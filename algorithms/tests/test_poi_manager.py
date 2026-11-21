"""Bernoulli manager distribution and rollout/PPO integration, on CPU."""

import itertools

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from algorithms.rmappo_jax import manager as manager_lib
from algorithms.rmappo_jax.mappo import gather_minibatch, unroll
from algorithms.rmappo_jax.network import evaluate_bernoulli, sample_bernoulli
from algorithms.rmappo_jax.trainer import make_train, validate
from algorithms.rmappo_jax.types import RMAPPOConfig
from environments.rover.config import load_rover_env


@pytest.fixture(autouse=True)
def _run_on_cpu():
    with jax.default_device(jax.devices("cpu")[0]):
        yield


def test_bernoulli_matches_joint_distribution_and_entropy():
    probabilities = np.array([0.2, 0.5, 0.8])
    masks = np.array(list(itertools.product([False, True], repeat=3)))
    joint = np.prod(np.where(masks, probabilities, 1 - probabilities), axis=-1)
    logits = jnp.asarray(np.log(probabilities / (1 - probabilities)))
    log_prob, entropy = evaluate_bernoulli(logits, jnp.asarray(masks))
    np.testing.assert_allclose(np.exp(log_prob), joint, rtol=1e-6)
    np.testing.assert_allclose(entropy, -np.sum(joint * np.log(joint)), rtol=1e-6)

    # The score of the joint action must include every POI's contribution.
    gradient = jax.grad(lambda x: evaluate_bernoulli(x, jnp.array([1, 0, 1]))[0])(logits)
    np.testing.assert_allclose(gradient, np.array([1, 0, 1]) - probabilities, atol=1e-6)


def test_bernoulli_sampling_and_deterministic_mode():
    logits = jnp.array([-2.0, 0.0, 2.0])
    mask, log_prob = sample_bernoulli(jax.random.PRNGKey(0), logits, True)
    np.testing.assert_array_equal(mask, [False, True, True])
    np.testing.assert_allclose(log_prob, evaluate_bernoulli(logits, mask)[0])

    masks, log_probs = jax.jit(lambda key: sample_bernoulli(
        key, jnp.broadcast_to(logits, (20000, 3)), False
    ))(jax.random.PRNGKey(1))
    assert masks.dtype == jnp.bool_
    assert log_probs.shape == (20000,)
    np.testing.assert_allclose(masks.mean(axis=0), jax.nn.sigmoid(logits), atol=0.015)
    np.testing.assert_allclose(
        log_probs, evaluate_bernoulli(logits, masks)[0], atol=1e-6
    )


def test_bernoulli_extreme_logits_have_finite_values_and_gradients():
    logits = jnp.array([-1000.0, 1000.0])
    # Include actions very unlikely under the policy.
    action = jnp.array([True, False])
    log_prob, entropy = evaluate_bernoulli(logits, action)
    np.testing.assert_allclose(log_prob, -2000.0)
    np.testing.assert_allclose(entropy, 0.0, atol=1e-6)
    gradient = jax.grad(lambda x: sum(evaluate_bernoulli(x, action)))(logits)
    assert np.isfinite(gradient).all()


@pytest.mark.parametrize("recurrent", [False, True])
def test_manager_with_16_pois_collects_updates_and_evaluates(recurrent):
    env = load_rover_env("rover_1a_16poi_c1_grid_gs", overrides=["env.max_steps=4"])
    config = RMAPPOConfig(
        use_poi_manager=True, use_recurrent_policy=recurrent,
        n_envs=2, n_steps=4, n_total_steps=8, n_epochs=1,
        n_minibatches=1, data_chunk_length=2, hidden_size=8, n_eval_episodes=2,
        manager_init_bias=0.0,
    )
    init_fn, collect_fn, update_fn, eval_fn, _ = make_train(config, env)
    state = init_fn(jax.random.PRNGKey(0))
    manager = state.train_state.manager
    head = manager.actor_ts.params["params"]["action_out"]
    assert head["kernel"].shape == (config.hidden_size, env.n_pois)
    np.testing.assert_array_equal(head["bias"], config.manager_init_bias)
    state, trajectory, last_value, _ = collect_fn(state)
    rover_traj, manager_traj = trajectory
    assert manager_traj.action.shape == (config.n_steps, config.n_envs, 1, 16)
    assert manager_traj.action.dtype == jnp.bool_
    assert manager_traj.log_prob.shape == (config.n_steps, config.n_envs, 1)
    np.testing.assert_array_equal(manager_traj.reward[..., 0], rover_traj.team_reward)

    # Re-evaluate stored masks through the actual PPO minibatch/unroll path.
    batch = gather_minibatch(
        manager_traj, jnp.ones_like(manager_traj.reward),
        jnp.arange(config.n_steps * config.n_envs), 1,
    )
    logits = unroll(
        manager.actor_ts.apply_fn, manager.actor_ts.params,
        batch["obs"], batch["actor_h0"], batch["mask"],
    )
    log_prob, entropy = evaluate_bernoulli(logits, batch["action"])
    np.testing.assert_allclose(log_prob, batch["old_log_prob"], atol=2e-6)
    updated, losses = update_fn(state, trajectory, last_value)
    assert all(np.isfinite(float(v)) for v in losses.values())
    np.testing.assert_allclose(losses["manager_ratio"], 1.0, atol=2e-6)
    np.testing.assert_allclose(losses["manager_dist_entropy"], entropy.mean(), atol=2e-6)
    np.testing.assert_allclose(
        losses["manager_pois_shown"], manager_traj.action.sum(axis=-1).mean()
    )
    before = jax.tree.leaves(manager.actor_ts.params)
    after = jax.tree.leaves(updated.train_state.manager.actor_ts.params)
    assert any(np.any(np.asarray(a) != np.asarray(b)) for a, b in zip(before, after))
    assert np.isfinite(float(eval_fn(updated.train_state, jax.random.PRNGKey(2))))


def _with_manager_bias(train_state, bias):
    """`train_state` with the manager head's bias set to `bias` `(P,)`. The
    head's kernel has gain 0.01, so the logits are about the bias."""
    manager = train_state.manager
    params = manager.actor_ts.params
    head = {**params["params"]["action_out"], "bias": jnp.asarray(bias, jnp.float32)}
    params = {**params, "params": {**params["params"], "action_out": head}}
    return train_state._replace(
        manager=manager._replace(actor_ts=manager.actor_ts.replace(params=params))
    )


@pytest.mark.parametrize("masks_reward", [False, True])
def test_manager_masks_the_reward_in_training_only(masks_reward):
    """Under `manager_masks_reward` a POI the manager hides pays nothing during
    collection, and both learners train on that gated reward. Evaluation stays
    ungated. Every POI's disc covers the arena, so both rovers meet every
    coupling on every step (each POI pays 0.1). The manager's head hides POI
    1 on every step (logit about -20)."""
    env = load_rover_env(
        "rover_2a_3poi_c2_implicit_gs",
        overrides=["env.max_steps=5", "env.pois.radius=30"],
    )
    config = RMAPPOConfig(
        use_poi_manager=True, manager_masks_reward=masks_reward,
        use_recurrent_policy=False, n_envs=2, n_steps=5, n_total_steps=10,
        n_epochs=1, n_minibatches=1, hidden_size=8, n_eval_episodes=2,
    )
    init_fn, collect_fn, _, eval_fn, _ = make_train(config, env)
    state = init_fn(jax.random.PRNGKey(0))
    state = state._replace(
        train_state=_with_manager_bias(state.train_state, [20.0, -20.0, 20.0])
    )
    _, (rover_traj, manager_traj), _, _ = collect_fn(state)
    np.testing.assert_array_equal(
        manager_traj.action[..., 0, :], [[[True, False, True]] * 2] * 5
    )
    # Gated, POI 1 pays nothing; ungated (negative control), all three pay.
    expected = 0.2 if masks_reward else 0.3
    np.testing.assert_allclose(rover_traj.team_reward, expected, rtol=1e-5)
    np.testing.assert_allclose(rover_traj.reward, expected, rtol=1e-5)
    np.testing.assert_allclose(manager_traj.reward[..., 0], rover_traj.team_reward)
    # Evaluation takes the same mask, which hides POI 1 only from the rovers:
    # 3 POIs x 0.1 x 5 steps either way.
    np.testing.assert_allclose(
        float(eval_fn(state.train_state, jax.random.PRNGKey(1))), 1.5, rtol=1e-5
    )


def test_manager_masks_reward_needs_the_manager():
    env = load_rover_env("rover_2a_3poi_c2_implicit_gs")
    config = RMAPPOConfig(
        n_envs=1, n_steps=1, n_total_steps=1, use_recurrent_policy=False,
        manager_masks_reward=True,
    )
    with pytest.raises(ValueError, match="needs use_poi_manager"):
        validate(config, env)


def test_manager_initial_show_bias():
    env = load_rover_env("rover_1a_16poi_c1_grid_gs")
    config = RMAPPOConfig(n_envs=1, n_steps=1, n_total_steps=1)
    manager = manager_lib.init_manager(jax.random.PRNGKey(0), config, env, 1)
    logits, _ = manager.actor_ts.apply_fn(
        manager.actor_ts.params, jnp.zeros((1, env.global_state_dim)),
        jnp.zeros(()), jnp.ones((1,)),
    )
    np.testing.assert_allclose(logits, config.manager_init_bias)
    mask, _, _ = manager_lib.manager_step(
        manager, jnp.zeros((1, env.global_state_dim)), jnp.ones((1,)),
        jax.random.PRNGKey(1), True,
    )
    assert np.asarray(mask).all()


def test_view_records_manager_masks_in_evaluation_order(tmp_path):
    """`view()`'s manager policy: the manager's most likely mask from the
    global state, the observation rebuilt under it, then the rovers' GRU, the
    mask of every step recorded in `Trajectory.poi_masks`."""
    import matplotlib

    matplotlib.use("Agg")
    from algorithms.rmappo_jax.mappo import zero_hidden
    from algorithms.rmappo_jax.run import ManagedViewPolicy
    from algorithms.rmappo_jax.trainer import actor_step
    from environments.rover.rollout import rollout_view_policy
    from environments.rover.visualize import plot_poi_masks, poi_shown_share, save_figure

    env = load_rover_env("rover_2a_3poi_c2_implicit_gs", overrides=["env.max_steps=5"])
    config = RMAPPOConfig(
        use_poi_manager=True, use_recurrent_policy=True, n_envs=1, n_steps=2,
        n_total_steps=2, data_chunk_length=2, hidden_size=8,
    )
    ts = make_train(config, env)[0](jax.random.PRNGKey(0)).train_state
    # POI 1 hidden, POIs 0 and 2 shown, at every step.
    ts = _with_manager_bias(ts, [3.0, -3.0, 3.0])

    key = jax.random.PRNGKey(4)
    traj = rollout_view_policy(env, ManagedViewPolicy(ts, env, config), key)
    n_steps = traj.actions.shape[0]
    assert n_steps == env.max_steps
    assert traj.poi_masks.shape == (n_steps, env.n_pois)
    assert traj.poi_masks.dtype == jnp.bool_
    np.testing.assert_array_equal(traj.poi_masks, [[True, False, True]] * n_steps)
    np.testing.assert_array_equal(poi_shown_share(traj), [1.0, 0.0, 1.0])

    # An independent host loop through the networks' apply functions.
    obs, state = env.reset(key)
    hidden = zero_hidden(config, (1, env.n_agents))
    for t in range(n_steps):
        logits, _ = ts.manager.actor_ts.apply_fn(
            ts.manager.actor_ts.params, env.global_state(state)[None],
            jnp.zeros(()), jnp.ones((1,)),
        )
        mask = logits[0] >= 0
        np.testing.assert_array_equal(traj.poi_masks[t], mask)
        masked = env.observe(state, mask)
        if t == 0:
            # Negative control: hiding POI 1 changes what the rovers sense.
            assert not np.allclose(masked, env.observe(state))
        action, _, hidden = actor_step(
            ts.actor_ts.apply_fn, ts.actor_ts.params, masked[None], hidden,
            jnp.ones((1,)), jax.random.PRNGKey(0), False, deterministic=True,
        )
        np.testing.assert_allclose(traj.actions[t], action[0], atol=1e-6)
        _, state, _, _, _, _ = env.step(state, action[0])

    path = save_figure(plot_poi_masks(env, traj, "test"), tmp_path / "masks.png")
    assert path.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"
