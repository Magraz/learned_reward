"""Learned POI manager for `rmappo_jax` (not part of on-policy).

Every step, before the rovers act, the manager reads the env's global state
(rover `env.global_state: true`: rover positions, POI positions, POI remaining
budgets) and picks which POIs every rover senses. Its action is a binary mask
sampled from one independent Bernoulli per POI, applied to all rovers through
`env.observe(state, poi_mask)`. A hidden POI still pays; only what the rovers
sense changes.

It is trained alongside the rovers, from the same rollouts, on the team reward,
with this stack's own PPO update (`mappo.ppo_update`): the manager is one
feedforward "agent" per env whose observation is the global state, with its
own actor, critic, Adams and ValueNorm, and every other setting the rovers'
(learning rates, epochs, clip, entropy, gamma, lambda, network size).

At initialization each POI's logit has bias `manager_init_bias`, so its show
probability starts near sigmoid(3) = 0.95 at the default 3.0 (all 3 POIs are
shown with probability about 0.86). The actor's output width is linear in the
number of POIs; no table of all possible masks is needed.
"""

import dataclasses

import jax
import jax.numpy as jnp
from flax.training.train_state import TrainState

from algorithms.mappo_jax.trainer import global_state_dim
from algorithms.rmappo_jax.mappo import RMAPPOTrainState, create_train_state
from algorithms.rmappo_jax.network import sample_bernoulli
from algorithms.rmappo_jax.types import RMAPPOConfig, Transition


def manager_config(config: RMAPPOConfig) -> RMAPPOConfig:
    """The rovers' config with a feedforward network (training chunks of 1)."""
    return dataclasses.replace(config, use_recurrent_policy=False)


def validate_manager(config: RMAPPOConfig, env):
    """Reject an env or config the manager cannot run on."""
    if not hasattr(env, "global_state"):
        raise ValueError(
            "use_poi_manager needs the env's global state: set "
            "`env.global_state: true` (rover) in the env group"
        )
    n_pois = getattr(env, "n_pois", None)
    if n_pois is None:
        raise ValueError("use_poi_manager needs an env with POIs (rover)")
    if config.use_proper_time_limits:
        raise NotImplementedError(
            "use_poi_manager with use_proper_time_limits is not implemented"
        )


def init_manager(rng, config: RMAPPOConfig, env, num_updates: int) -> RMAPPOTrainState:
    """The manager's networks, optimizers and ValueNorm, its head biased
    toward showing POIs (`manager_init_bias` on each Bernoulli logit)."""
    n_pois = env.n_pois
    gs_dim = global_state_dim(env)
    state = create_train_state(
        rng, manager_config(config), gs_dim, gs_dim, n_pois, True, num_updates,
        n_critic_outputs=1,
    )
    params = state.actor_ts.params
    head = {
        **params["params"]["action_out"],
        "bias": jnp.full((n_pois,), config.manager_init_bias, dtype=jnp.float32),
    }
    params = {**params, "params": {**params["params"], "action_out": head}}
    # A fresh TrainState so the optimizer state matches the new params tree;
    # Adam's initial state does not depend on the parameter values.
    actor_ts = TrainState.create(
        apply_fn=state.actor_ts.apply_fn, params=params, tx=state.actor_ts.tx
    )
    return state._replace(actor_ts=actor_ts)


def manager_value(manager: RMAPPOTrainState, global_state, episode_mask):
    """`(B, 1)` normalized value of each env's global state `(B, gs_dim)`."""
    value, _ = manager.critic_ts.apply_fn(
        manager.critic_ts.params, global_state, jnp.zeros(()), episode_mask
    )
    return value


def manager_step(
    manager: RMAPPOTrainState, global_state, episode_mask, rng, deterministic
):
    """One decision per env from its global state `(B, gs_dim)`.

    Returns `(poi_mask (B, P), log_prob (B,), value (B, 1))`;
    `deterministic` shows each POI whose logit is nonnegative (evaluation).
    """
    logits, _ = manager.actor_ts.apply_fn(
        manager.actor_ts.params, global_state, jnp.zeros(()), episode_mask
    )
    poi_mask, log_prob = sample_bernoulli(rng, logits, deterministic)
    value = manager_value(manager, global_state, episode_mask)
    return poi_mask, log_prob, value


def manager_transition(
    global_state, episode_mask, poi_mask, log_prob, value, team_reward, done
) -> Transition:
    """The manager's step as a `Transition` with one agent per env, which
    `mappo.ppo_update` trains like the rovers' (feedforward placeholders for
    the hidden states, no action mask). It is paid the team reward."""
    ones = jnp.ones((global_state.shape[0], 1))
    return Transition(
        obs=global_state[:, None, :],
        global_state=global_state,
        actor_h=jnp.zeros(()),
        critic_h=jnp.zeros(()),
        mask=episode_mask,
        action=poi_mask[:, None, :],
        log_prob=log_prob[:, None],
        value=value,
        reward=team_reward[:, None],
        done=done,
        active_mask=ones,
        action_mask=jnp.zeros(()),
        team_reward=team_reward,
    )


def pois_shown(manager_trajectory: Transition) -> jax.Array:
    """Mean number of POIs the manager showed per step over a rollout."""
    return manager_trajectory.action.sum(axis=-1).mean()
