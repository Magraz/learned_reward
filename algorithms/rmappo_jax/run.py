"""Runner for `rmappo_jax`, on-policy's MAPPO over the functional JAX envs.

Subclasses `MAPPO_JAX_Runner`, so env construction (`make_env`), results
layout, the train loop, stats tracking, checkpoint cadence and `evaluate()` are
inherited. This class supplies the config, the train functions, the checkpoint
trees (which add the ValueNorm state) and the recurrent `view()` policy.

Resume (`checkpoint=true`) restores the networks, both Adam states, ValueNorm
and the RNG chains. The rollout itself is not saved: a resumed run starts fresh
episodes with zeroed hidden states.
"""

from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from flax.serialization import from_bytes, msgpack_restore, to_bytes

from algorithms.mappo_jax.run import MAPPO_JAX_Runner, make_base_env, make_env
from algorithms.mappo_jax.trainer import global_state_dim, global_state_fn
from algorithms.rmappo_jax.manager import init_manager, observe_with_manager
from algorithms.rmappo_jax.mappo import (
    RMAPPOTrainState,
    create_train_state,
    denormalize_values,
    zero_hidden,
)
from algorithms.rmappo_jax.trainer import (
    actor_step,
    make_train,
    n_critic_outputs,
    validate,
)
from algorithms.rmappo_jax.types import Experiment, Model_Params, Params, make_config

# Settings that leave every parameter shape unchanged, so a checkpoint alone
# cannot tell which values trained it. They are stored with the train
# checkpoint and a resume under other values raises (cf. the scheduler record
# in simplified_feudal_mappo_jax/run.py).
META_FIELDS = (
    "n_envs", "n_steps", "n_epochs", "n_minibatches", "lr", "critic_lr",
    "opti_eps", "gamma", "gae_lambda", "eps_clip", "ent_coef", "val_coef",
    "use_max_grad_norm", "grad_clip", "use_valuenorm", "use_huber_loss",
    "huber_delta", "use_clipped_value_loss", "use_value_active_masks",
    "use_policy_active_masks", "use_proper_time_limits", "use_linear_lr_decay",
    "data_chunk_length", "use_ReLU", "gain",
)


class RecurrentViewPolicy:
    """`view()`'s policy, carrying the GRU hidden state across the decisions of
    one rendered episode (the `StatelessViewPolicy` interface, including its
    `deterministic` and `seed`)."""

    def __init__(self, train_state, env, config, deterministic=True, seed=0):
        discrete = getattr(env, "discrete", False)
        n_agents, obs_dim = env.n_agents, env.observation_dim
        self._initial_hidden = zero_hidden(config, (1, n_agents))

        @jax.jit
        def act(obs, hidden, mask, key):
            key, action_key = jax.random.split(key)
            actions, _, hidden = actor_step(
                train_state.actor_ts.apply_fn,
                train_state.actor_ts.params,
                obs.reshape(1, n_agents, obs_dim),
                hidden,
                jnp.ones((1,)),
                action_key,
                discrete,
                deterministic=deterministic,
                action_mask=None if mask is None else mask.reshape(1, n_agents, -1),
            )
            return actions[0], hidden, key

        self._act = act
        self._key = jax.random.PRNGKey(seed)
        self.reset()

    def reset(self):
        self._hidden = self._initial_hidden

    def __call__(self, obs, mask=None, state=None):
        actions, self._hidden, self._key = self._act(
            obs, self._hidden, mask, self._key
        )
        return actions


class ManagedViewPolicy:
    """`view()`'s policy with the POI manager, in evaluation's order.

    Every decision, the manager picks a mask from the global state of `state`,
    the rovers' observation is rebuilt under that mask, and the rovers'
    `RecurrentViewPolicy` acts on it. Deterministic (default), the manager
    takes its most likely mask (a POI is shown when its logit is nonnegative);
    sampled, it draws one, from its own key chain (`seed` folded with 1, as
    training folds the rovers' action key). The decision's mask `(P,)` is left
    in `last_poi_mask`, which `rollout_view_policy` records.
    """

    def __init__(self, train_state, env, config, deterministic=True, seed=0):
        self._rovers = RecurrentViewPolicy(
            train_state, env, config, deterministic, seed
        )
        global_state = global_state_fn(env)
        v_observe = jax.vmap(env.observe)

        @jax.jit
        def manage(obs, state, key):
            key, manager_key = jax.random.split(key)
            batched = jax.tree.map(lambda x: x[None], state)
            obs, (poi_mask, _, _) = observe_with_manager(
                train_state.manager,
                v_observe,
                global_state(obs[None], batched),
                jnp.ones((1,)),
                batched,
                manager_key,
                deterministic=deterministic,
            )
            return obs[0], poi_mask[0], key

        self._manage = manage
        self._key = jax.random.fold_in(jax.random.PRNGKey(seed), 1)
        self.reset()

    def reset(self):
        self._rovers.reset()
        self.last_poi_mask = None

    def __call__(self, obs, mask=None, state=None):
        if state is None:
            raise ValueError("the POI manager's view policy needs the env state")
        obs, self.last_poi_mask, self._key = self._manage(obs, state, self._key)
        return self._rovers(obs, mask)


class RMAPPO_JAX_Runner(MAPPO_JAX_Runner):
    def __init__(
        self,
        device: str,
        batch_dir: Path,
        trials_dir: Path,
        trial_id: str,
        checkpoint: bool,
        exp_config: Experiment,
        env_config: dict,
    ):
        self.params = Params(**exp_config.params)
        self.model_params = Model_Params(**exp_config.model_params)
        self.config = make_config(
            self.params, self.model_params, env_config.get("n_envs")
        )
        # Built and checked before the results directory exists, so a rejected
        # configuration leaves nothing behind.
        self.env = make_env(env_config)
        validate(self.config, self.env)
        self.base_env = make_base_env(env_config, self.env)
        self._init_trial(
            device, batch_dir, trials_dir, trial_id, checkpoint,
            self.params.random_seeds,
        )

        config = self.config
        if config.n_steps != self.params.n_steps:
            print(
                f"  n_steps {self.params.n_steps} -> {config.n_steps} "
                f"(a multiple of data_chunk_length={config.data_chunk_length})"
            )
        policy = (
            f"rMAPPO (GRU, chunks of {config.data_chunk_length})"
            if config.use_recurrent_policy
            else "MAPPO (feedforward)"
        )
        print(
            f"{policy} (JAX) | env={env_config.get('environment')} | "
            f"n_agents={self.env.n_agents} | n_envs={config.n_envs} | "
            f"n_steps={config.n_steps} | hidden={config.hidden_size} | "
            f"epochs={config.n_epochs} x minibatches={config.n_minibatches} | "
            f"total={config.n_total_steps} | backend={jax.default_backend()}"
        )
        print(
            f"  valuenorm={config.use_valuenorm} huber={config.use_huber_loss} "
            f"clipped_value={config.use_clipped_value_loss} "
            f"proper_time_limits={config.use_proper_time_limits} "
            f"lr_decay={config.use_linear_lr_decay} | "
            f"reward_mode={getattr(self.env, 'reward_mode', 'team')} | "
            f"centralized input: gs_dim={global_state_dim(self.env)} "
            f"(env hook: {hasattr(self.env, 'global_state')})"
        )
        if config.use_poi_manager:
            print(
                f"  poi_manager=on | {self.env.n_pois} Bernoulli show/hide outputs "
                f"| init logit bias {config.manager_init_bias} per POI "
                f"| masks reward: "
                + (
                    "on (training only; evaluation ungated)"
                    if config.manager_masks_reward
                    else "off"
                )
            )
        self._print_base_env()

    def _make_train(self, env=None):
        return make_train(self.config, self.env if env is None else env)

    def _eval_every(self) -> int:
        return self.config.eval_interval

    def _view_policy(self, train_state, deterministic=True, seed=0):
        policy = (
            ManagedViewPolicy if self.config.use_poi_manager else RecurrentViewPolicy
        )
        return policy(train_state, self.env, self.config, deterministic, seed)

    def _critic_apply(self, train_state):
        """The GRU (or feedforward) critic, ValueNorm undone
        (`MAPPO_JAX_Runner._critic_apply` gives the interface). `h` is one
        hidden state per env `(B, recurrent_N, H)`, as in training."""
        config = self.config
        critic_ts, value_norm = train_state.critic_ts, train_state.value_norm

        def apply(global_state, h, mask):
            values, h = critic_ts.apply_fn(critic_ts.params, global_state, h, mask)
            return denormalize_values(value_norm, values, config), h

        return apply, lambda batch: zero_hidden(config, (batch,))

    # ------------------------------------------------------------------ io

    @staticmethod
    def _params_tree(train_state: RMAPPOTrainState) -> dict:
        tree = {
            "actor": train_state.actor_ts.params,
            "critic": train_state.critic_ts.params,
            "value_norm": train_state.value_norm,
        }
        manager = train_state.manager
        if manager is not None:
            tree["manager"] = {
                "actor": manager.actor_ts.params,
                "critic": manager.critic_ts.params,
                "value_norm": manager.value_norm,
            }
        return tree

    def save_params(self, train_state, path):
        with open(path, "wb") as f:
            f.write(to_bytes(self._params_tree(train_state)))

    def _config_meta(self) -> np.ndarray:
        return np.array(
            [float(getattr(self.config, name)) for name in META_FIELDS], np.float64
        )

    def _checkpoint_tree(self, runner_state, eval_rng) -> dict:
        ts = runner_state.train_state
        tree = {
            "actor_ts": ts.actor_ts,
            "critic_ts": ts.critic_ts,
            "value_norm": ts.value_norm,
            "rng": runner_state.rng,
            "eval_rng": eval_rng,
            "config_meta": self._config_meta(),
        }
        if ts.manager is not None:
            tree["manager"] = {
                "actor_ts": ts.manager.actor_ts,
                "critic_ts": ts.manager.critic_ts,
                "value_norm": ts.manager.value_norm,
            }
        return tree

    def _check_config_meta(self, raw: dict, path):
        meta, stored = self._config_meta(), raw.get("config_meta")
        if stored is None:
            raise ValueError(f"{path} has no rmappo_jax config record")
        stored = np.asarray(stored, np.float64)
        if stored.shape != meta.shape or not np.array_equal(stored, meta):
            changed = {
                name: (old, new)
                for name, old, new in zip(META_FIELDS, stored.tolist(), meta.tolist())
                if old != new
            } if stored.shape == meta.shape else stored.tolist()
            raise ValueError(
                f"{path} was trained under other settings ((checkpoint, this "
                f"config): {changed}). Resume with the settings that trained it, "
                "or train into another model group."
            )

    def _save_train_checkpoint(self, runner_state, eval_rng, path):
        with open(path, "wb") as f:
            f.write(to_bytes(self._checkpoint_tree(runner_state, eval_rng)))

    def _load_train_checkpoint(self, runner_state, eval_rng, path):
        with open(path, "rb") as f:
            data = f.read()
        self._check_config_meta(msgpack_restore(data), path)
        loaded = from_bytes(self._checkpoint_tree(runner_state, eval_rng), data)
        print(f"Train state loaded from {path} (episodes restart from reset)")
        manager = None
        if "manager" in loaded:
            manager = RMAPPOTrainState(
                actor_ts=loaded["manager"]["actor_ts"],
                critic_ts=loaded["manager"]["critic_ts"],
                value_norm=loaded["manager"]["value_norm"],
            )
        train_state = RMAPPOTrainState(
            actor_ts=loaded["actor_ts"],
            critic_ts=loaded["critic_ts"],
            value_norm=loaded["value_norm"],
            manager=manager,
        )
        return (
            runner_state._replace(train_state=train_state, rng=loaded["rng"]),
            loaded["eval_rng"],
        )

    def _load_train_state(self) -> RMAPPOTrainState:
        """A train state with the networks and ValueNorm of the latest save."""
        train_state = create_train_state(
            jax.random.PRNGKey(0),
            self.config,
            self.env.observation_dim,
            global_state_dim(self.env),
            self.env.action_dim,
            getattr(self.env, "discrete", False),
            num_updates=1,
            # Must match training, or the restored critic head won't fit.
            n_critic_outputs=n_critic_outputs(self.env),
        )
        if self.config.use_poi_manager:
            train_state = train_state._replace(
                manager=init_manager(
                    jax.random.PRNGKey(0), self.config, self.env, num_updates=1
                )
            )
        path = self.dirs["models"] / "models_finished.msgpack"
        if not path.exists():
            path = self.dirs["models"] / "models_checkpoint.msgpack"
        with open(path, "rb") as f:
            params = from_bytes(self._params_tree(train_state), f.read())
        print(f"Params loaded from {path}")

        def restore(template: RMAPPOTrainState, saved: dict) -> RMAPPOTrainState:
            return RMAPPOTrainState(
                actor_ts=template.actor_ts.replace(params=saved["actor"]),
                critic_ts=template.critic_ts.replace(params=saved["critic"]),
                value_norm=saved["value_norm"],
            )

        restored = restore(train_state, params)
        if train_state.manager is not None:
            restored = restored._replace(
                manager=restore(train_state.manager, params["manager"])
            )
        return restored
