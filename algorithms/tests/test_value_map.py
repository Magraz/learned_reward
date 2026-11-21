"""Critic value maps (`environments/rover/value_map.py`) and the `view()` code
that draws them, on CPU with untrained networks."""

import dataclasses

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from algorithms.mappo_jax.trainer import global_state_dim
from algorithms.rmappo_jax import value_norm as vn_lib
from algorithms.rmappo_jax.mappo import create_train_state
from algorithms.rmappo_jax.run import RMAPPO_JAX_Runner
from algorithms.rmappo_jax.trainer import make_train, n_critic_outputs
from algorithms.rmappo_jax.types import RMAPPOConfig
from environments.rover.config import load_rover_env
from environments.rover.rollout import rollout_view_policy
from environments.rover.value_map import (
    CriticProbe,
    discounted_return_to_go,
    grid_axes,
    phase_steps,
    state_at,
    visitation_maps,
)


@pytest.fixture(autouse=True)
def _run_on_cpu():
    with jax.default_device(jax.devices("cpu")[0]):
        yield


def _stub(cls, env, config, train_state=None, tmp_path=None):
    """A runner without a results directory: the attributes `view()` reads,
    and `_load_train_state` returning `train_state`."""

    class Stub(cls):
        def _load_train_state(self):
            return train_state

    runner = object.__new__(Stub)
    runner.env, runner.config = env, config
    if tmp_path is not None:
        runner.dirs = {"logs": tmp_path / "logs"}
        runner.trial_dir = tmp_path / "model_group" / "0"
        runner.trial_id, runner.rng_seed = "0", 7
    return runner


def _rmappo(env, recurrent, seed=9):
    config = RMAPPOConfig(
        n_envs=1, n_steps=10, n_total_steps=10, hidden_size=8,
        use_recurrent_policy=recurrent,
    )
    ts = create_train_state(
        jax.random.PRNGKey(seed), config, env.observation_dim, global_state_dim(env),
        env.action_dim, False, 1, n_critic_outputs=n_critic_outputs(env),
    )
    # Non-trivial ValueNorm statistics, so the denormalization is exercised.
    value_norm = ts.value_norm
    for returns in (jnp.array([1.0, 3.0]), jnp.array([2.0, 6.0])):
        value_norm = vn_lib.update(value_norm, returns)
    return config, ts._replace(value_norm=value_norm)


def _episode(env, runner, ts, reset_seed=1, **policy):
    """One `rollout_view_policy` episode from `PRNGKey(reset_seed)`; `policy`
    goes to `_view_policy` (`deterministic`, `seed`)."""
    return rollout_view_policy(
        env, runner._view_policy(ts, **policy), jax.random.PRNGKey(reset_seed)
    )


# ------------------------------------------------------------- along the path


@pytest.mark.parametrize(
    "group, overrides",
    [
        ("rover_4a_2poi_c2", []),
        ("rover_2a_3poi_c2_implicit_gs", []),
        ("rover_4a_2poi_c2", ["env.reward.mode=difference_rewards"]),
    ],
    ids=["sectors", "global_state", "per_rover_head"],
)
@pytest.mark.parametrize("recurrent", [False, True], ids=["mappo_mlp", "rmappo"])
def test_critic_along_matches_an_independent_host_loop(group, overrides, recurrent):
    env = load_rover_env(group, ["env.max_steps=12", *overrides])
    config, ts = _rmappo(env, recurrent)
    runner = _stub(RMAPPO_JAX_Runner, env, config)
    traj = _episode(env, runner, ts)
    probe = CriticProbe(env, *runner._view_critic(ts))
    values, _ = probe.along(traj)

    # The critic stepped by hand: its input built from the state, its hidden
    # state carried, ValueNorm undone, a per-rover head averaged.
    hidden = jnp.zeros((1, config.recurrent_N, config.hidden_size)) if recurrent else jnp.zeros(())
    expected = []
    for t in range(traj.positions.shape[0]):
        state = state_at(traj, t)
        if hasattr(env, "global_state"):
            gs = env.global_state(state)[None]
        else:
            gs = env.observe(state).reshape(1, -1)
        raw, hidden = ts.critic_ts.apply_fn(ts.critic_ts.params, gs, hidden, jnp.ones((1,)))
        expected.append(float(vn_lib.denormalize(ts.value_norm, raw).mean()))
    np.testing.assert_allclose(values, expected, rtol=1e-5, atol=1e-5)

    if recurrent:
        # Negative control: without the carried hidden state the values change.
        memoryless = []
        for t in range(traj.positions.shape[0]):
            state = state_at(traj, t)
            gs = (env.global_state(state)[None] if hasattr(env, "global_state")
                  else env.observe(state).reshape(1, -1))
            raw, _ = ts.critic_ts.apply_fn(
                ts.critic_ts.params, gs, jnp.zeros_like(hidden), jnp.ones((1,))
            )
            memoryless.append(float(vn_lib.denormalize(ts.value_norm, raw).mean()))
        assert not np.allclose(values[1:], memoryless[1:], atol=1e-4)


def test_critic_along_reuses_one_compilation_across_episode_lengths():
    env = load_rover_env("rover_4a_2poi_c2", ["env.max_steps=12"])
    config, ts = _rmappo(env, True)
    runner = _stub(RMAPPO_JAX_Runner, env, config)
    probe = CriticProbe(env, *runner._view_critic(ts))
    traj = _episode(env, runner, ts)
    # A 4-step episode: its first 5 states.
    short = dataclasses.replace(
        traj, states=jax.tree.map(lambda x: x[:5], traj.states),
        positions=traj.positions[:5], poi_remaining=traj.poi_remaining[:5],
        rewards=traj.rewards[:4],
    )
    values, _ = probe.along(traj)
    short_values, _ = probe.along(short)
    np.testing.assert_allclose(short_values, values[:5], rtol=1e-6)
    assert probe._along._cache_size() == 1


# ------------------------------------------------------------------- slices


@pytest.mark.parametrize("group", ["rover_4a_2poi_c2", "rover_2a_3poi_c2_implicit_gs"])
@pytest.mark.parametrize("recurrent", [False, True], ids=["mappo_mlp", "rmappo"])
def test_a_slice_at_the_team_centre_is_the_value_along_the_path(group, recurrent):
    env = load_rover_env(group, ["env.max_steps=12"])
    config, ts = _rmappo(env, recurrent)
    runner = _stub(RMAPPO_JAX_Runner, env, config)
    traj = _episode(env, runner, ts)
    value_fn, initial_hidden = runner._view_critic(ts)
    values, hidden = CriticProbe(env, value_fn, initial_hidden).along(traj)
    probe = CriticProbe(env, value_fn, initial_hidden, n_headings=1)
    for t in (0, 6, 11):
        state = state_at(traj, t)
        centre = np.asarray(state.rovers.position).mean(axis=0)
        here = probe.values_at(state, jax.tree.map(lambda x: x[t], hidden), centre[None])
        np.testing.assert_allclose(here[0], values[t], rtol=1e-5, atol=1e-5)


def test_moving_the_team_changes_a_global_state_critic():
    env = load_rover_env("rover_2a_3poi_c2_implicit_gs", ["env.max_steps=12"])
    config, ts = _rmappo(env, False)
    runner = _stub(RMAPPO_JAX_Runner, env, config)
    traj = _episode(env, runner, ts)
    probe = CriticProbe(env, *runner._view_critic(ts))
    xs, ys = grid_axes(env, 10)
    grid = probe.team_slice(state_at(traj, 0), jnp.zeros(()), xs, ys)
    assert grid.shape == (len(ys), len(xs))
    assert np.ptp(grid) > 1e-3


def test_a_global_state_critic_ignores_the_heading():
    env = load_rover_env("rover_2a_3poi_c2_implicit_gs", ["env.max_steps=12"])
    config, ts = _rmappo(env, True)
    runner = _stub(RMAPPO_JAX_Runner, env, config)
    traj = _episode(env, runner, ts)
    value_fn, initial_hidden = runner._view_critic(ts)
    assert CriticProbe(env, value_fn, initial_hidden).n_headings == 1
    _, hidden = CriticProbe(env, value_fn, initial_hidden).along(traj)
    xs, ys = grid_axes(env, 8)
    state, h = state_at(traj, 5), jax.tree.map(lambda x: x[5], hidden)
    one = CriticProbe(env, value_fn, initial_hidden, n_headings=1).team_slice(state, h, xs, ys)
    many = CriticProbe(env, value_fn, initial_hidden, n_headings=16).team_slice(state, h, xs, ys)
    np.testing.assert_allclose(one, many, rtol=1e-5, atol=1e-6)


def test_the_team_moves_as_a_block_inside_the_arena():
    """Stub critics that return what the slice built: the team keeps its
    formation and centres on the cell, positions clipped to the arena, and
    every rover's heading gets the same offset."""
    env = load_rover_env("rover_4a_2poi_c2")
    _, state = env.reset(jax.random.PRNGKey(3))
    width = env.world_size[0]
    points = np.array([[5.0, 5.0], [0.0, 0.0], [width, width]])

    def probe(read):
        return CriticProbe(env, lambda s, h, m: (read(s)[:, None], h),
                           lambda b: jnp.zeros(()), n_headings=4)

    def at(read, pts):
        return probe(read).values_at(state, jnp.zeros(()), pts)

    np.testing.assert_allclose(
        at(lambda s: s.rovers.position[..., 0].mean(-1), points[:1]), [5.0], rtol=1e-6
    )
    assert np.all(at(lambda s: s.rovers.position.min((1, 2)), points) >= 0.0)
    assert np.all(at(lambda s: s.rovers.position.max((1, 2)), points) <= width)
    # The relative heading of two rovers is the same at every offset.
    gap = float(state.rovers.heading[0] - state.rovers.heading[1])
    relative = at(lambda s: jnp.cos(s.rovers.heading[:, 0] - s.rovers.heading[:, 1]),
                  points[:1])
    np.testing.assert_allclose(relative, [np.cos(gap)], atol=1e-5)
    # ... while the headings themselves turn through the offsets.
    np.testing.assert_allclose(
        at(lambda s: jnp.cos(s.rovers.heading[:, 0]), points[:1]), [0.0], atol=1e-5
    )


# --------------------------------------------------- returns, phases, binning


def test_discounted_return_to_go():
    returns = discounted_return_to_go([1.0, 0.0, 2.0], 0.5)
    np.testing.assert_allclose(returns, [1.0 + 0.25 * 2.0, 1.0, 2.0, 0.0])


class _Budgets:
    """The two fields `phase_steps` reads."""

    def __init__(self, remaining):
        self.poi_remaining = np.asarray(remaining, np.float32)
        self.rewards = np.zeros(len(remaining) - 1)


def _budgets(n_steps, used_at, n_pois):
    remaining = np.ones((n_steps + 1, n_pois))
    for poi, t in used_at.items():
        remaining[t:, poi] = 0.0
    return _Budgets(remaining)


def test_phase_steps():
    # Two POIs used up mid-episode, one on the final step (no panel: the
    # state after the last step is never evaluated).
    phases = phase_steps(_budgets(100, {0: 20, 1: 45, 2: 100}, 3))
    assert phases == [(0, "start"), (20, "P0 used up"), (45, "P1 used up")]
    # Two POIs used up on the same step share a panel.
    phases = phase_steps(_budgets(100, {0: 30, 1: 30, 2: 60}, 3))
    assert phases == [(0, "start"), (30, "P0 + P1 used up"), (60, "P2 used up")]
    # None used up: evenly spaced steps fill to three panels.
    phases = phase_steps(_budgets(100, {}, 3))
    assert [t for t, _ in phases] == [0, 50, 99]
    # Many used up: subsampled evenly to six panels, step 0 and the first and
    # last depletion kept.
    phases = phase_steps(_budgets(100, {p: 5 * (p + 1) for p in range(16)}, 16))
    steps = [t for t, _ in phases]
    assert len(steps) == 6 and steps[0] == 0 and steps[1] == 5 and steps[-1] == 80
    assert steps == sorted(set(steps))


def test_visitation_maps_match_a_numpy_reference():
    env = load_rover_env("rover_4a_2poi_c2")
    rng = np.random.default_rng(0)
    centroids = rng.uniform(0, 10, (500, 2))
    centroids[:50] = [2.1, 7.9]  # one crowded cell
    values, returns = rng.normal(size=500), rng.normal(size=500)
    maps = visitation_maps(env, centroids, values, returns, n_cells=5, min_visits=3)
    assert maps.visits.sum() == 500
    assert maps.visits.shape == (5, 5)
    # Reference: every cell's mean by direct masking (cells are 2 x 2 units).
    col = np.minimum((centroids[:, 0] // 2).astype(int), 4)
    row = np.minimum((centroids[:, 1] // 2).astype(int), 4)
    for r in range(5):
        for c in range(5):
            inside = (row == r) & (col == c)
            assert maps.visits[r, c] == inside.sum()
            if inside.sum() >= 3:
                np.testing.assert_allclose(maps.mean_value[r, c], values[inside].mean())
                np.testing.assert_allclose(maps.mean_return[r, c], returns[inside].mean())
            else:
                assert np.isnan(maps.mean_value[r, c])


# ------------------------------------------------------------ view policies


@pytest.mark.parametrize("recurrent", [False, True], ids=["mappo_mlp", "rmappo"])
def test_sampled_view_policies(recurrent):
    env = load_rover_env("rover_4a_2poi_c2", ["env.max_steps=8"])
    config, ts = _rmappo(env, recurrent)
    runner = _stub(RMAPPO_JAX_Runner, env, config)
    default = _episode(env, runner, ts)
    deterministic = _episode(env, runner, ts, deterministic=True, seed=5)
    np.testing.assert_array_equal(default.actions, deterministic.actions)
    first = _episode(env, runner, ts, deterministic=False, seed=5)
    again = _episode(env, runner, ts, deterministic=False, seed=5)
    other = _episode(env, runner, ts, deterministic=False, seed=6)
    np.testing.assert_array_equal(first.actions, again.actions)
    assert not np.allclose(first.actions, default.actions)
    assert not np.allclose(first.actions, other.actions)
    # The key chain runs on across episodes of one sampled policy.
    policy = runner._view_policy(ts, deterministic=False, seed=5)
    one = rollout_view_policy(env, policy, jax.random.PRNGKey(1))
    two = rollout_view_policy(env, policy, jax.random.PRNGKey(1))
    assert not np.allclose(one.actions, two.actions)


# ------------------------------------------------------------- end to end


def _pngs(paths, names):
    for name in names:
        matching = [p for p in paths if p.name == name]
        assert matching, f"{name} not written"
        assert matching[0].read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"


@pytest.mark.parametrize(
    "group",
    ["rover_1a_3poi_c1_implicit012_ord_hidden", "rover_4a_2poi_c2",
     "rover_2a_3poi_c2_implicit_gs"],
)
@pytest.mark.parametrize("recurrent", [False, True], ids=["mappo_mlp", "rmappo"])
def test_rmappo_view_writes_the_value_maps(group, recurrent, tmp_path):
    import matplotlib

    matplotlib.use("Agg")
    env = load_rover_env(group, ["env.max_steps=10"])
    config, ts = _rmappo(env, recurrent)
    runner = _stub(RMAPPO_JAX_Runner, env, config, ts, tmp_path)
    paths = runner._view_rover(n_episodes=1, n_visitation_episodes=3)
    _pngs(paths, ["value_map_seed8.png", "value_visitation.png"])


def test_manager_view_writes_the_value_maps(tmp_path):
    import matplotlib

    matplotlib.use("Agg")
    env = load_rover_env("rover_2a_3poi_c2_implicit_gs", ["env.max_steps=10"])
    config = RMAPPOConfig(
        use_poi_manager=True, use_recurrent_policy=True, n_envs=1, n_steps=2,
        n_total_steps=2, data_chunk_length=2, hidden_size=8,
    )
    ts = make_train(config, env)[0](jax.random.PRNGKey(0)).train_state
    runner = _stub(RMAPPO_JAX_Runner, env, config, ts, tmp_path)
    paths = runner._view_rover(n_episodes=1, n_visitation_episodes=3)
    _pngs(paths, ["manager_masks_seed8.png", "value_map_seed8.png",
                  "value_visitation.png"])
    # Sampling, the manager draws its masks too: not always the most likely.
    sampled = rollout_view_policy(
        env, runner._view_policy(ts, deterministic=False, seed=3), jax.random.PRNGKey(2)
    )
    likely = rollout_view_policy(env, runner._view_policy(ts), jax.random.PRNGKey(2))
    assert not np.array_equal(sampled.poi_masks, likely.poi_masks)


def test_mappo_jax_view_writes_the_value_maps(tmp_path):
    import matplotlib

    matplotlib.use("Agg")
    from algorithms.mappo_jax.mappo import create_train_state as mappo_train_state
    from algorithms.mappo_jax.run import MAPPO_JAX_Runner
    from algorithms.mappo_jax.types import MAPPOConfig

    env = load_rover_env("rover_4a_2poi_c2", ["env.max_steps=10"])
    config = MAPPOConfig(n_envs=1, n_steps=10, hidden_dim=8)
    ts = mappo_train_state(
        jax.random.PRNGKey(0), config, env.observation_dim, global_state_dim(env),
        env.action_dim, False,
    )
    runner = _stub(MAPPO_JAX_Runner, env, config, ts, tmp_path)
    # Its critic: no hidden state, no value normalization.
    value_fn, initial_hidden = runner._view_critic(ts)
    traj = _episode(env, runner, ts)
    values, _ = CriticProbe(env, value_fn, initial_hidden).along(traj)
    gs = env.observe(state_at(traj, 3)).reshape(1, -1)
    np.testing.assert_allclose(
        values[3], float(ts.critic_ts.apply_fn(ts.critic_ts.params, gs)[0]), rtol=1e-5
    )
    paths = runner._view_rover(n_episodes=1, n_visitation_episodes=3)
    _pngs(paths, ["value_map_seed8.png", "value_visitation.png"])
