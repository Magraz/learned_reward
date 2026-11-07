"""Tests for the pure-JAX rover domain (environments/rover).

Pinned to CPU: the env is tiny, and sharing a GPU with a training job turns
cuSolver / OOM errors into failures that read like assertion errors.
"""

import dataclasses
import math
import warnings

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from environments.rover import (
    BaseRoverEnv,
    CouplingDepletionReward,
    ObservationModel,
    POIModel,
    RewardModel,
    RewardOutput,
    RoverEnvState,
    RoverState,
    SectorDensityObservation,
    StaticPOIs,
)
from environments.rover.config import (
    list_variants,
    load_env_config,
    load_rover_env,
    make_rover_env,
)
from environments.rover.policies import (
    ROUTE_PLANS,
    entry_steps,
    make_scripted_coalition_policy,
    plan_routes,
    planned_rollouts,
    random_policy,
)
from environments.rover.pois import order_reached, unlocked
from environments.rover.rollout import rollout, rollout_reset
from environments.rover.scripted_steps import solve_steps


@pytest.fixture(autouse=True)
def _run_on_cpu():
    with jax.default_device(jax.devices("cpu")[0]):
        yield


def make_env(n_agents=2, entries=None, observation=None, reward=None, **kwargs):
    entries = entries or [{"position": [5.0, 5.0], "coupling": 2}]
    return BaseRoverEnv(
        n_agents,
        StaticPOIs(entries, radius=1.0, value=1.0, rate=0.05),
        observation,
        reward,
        world_size=(10.0, 10.0),
        **kwargs,
    )


def state_at(env, positions, headings=None, remaining=None):
    _, state = env.reset(jax.random.PRNGKey(0))
    positions = jnp.asarray(positions, dtype=jnp.float32)
    headings = (
        jnp.zeros(positions.shape[0]) if headings is None
        else jnp.asarray(headings, dtype=jnp.float32)
    )
    pois = state.pois
    if remaining is not None:
        pois = dataclasses.replace(pois, remaining=jnp.asarray(remaining, jnp.float32))
    return RoverEnvState(RoverState(positions, headings), pois, state.t)


def stay(n_agents):
    """No turn, zero speed."""
    return jnp.tile(jnp.array([0.0, -1.0]), (n_agents, 1))


# ---------------------------------------------------------------- interface


def test_reset_and_step_jit_and_vmap():
    env = load_rover_env("rover_2poi_c2")
    keys = jax.random.split(jax.random.PRNGKey(0), 8)
    obs, state = jax.jit(jax.vmap(env.reset))(keys)
    assert obs.shape == (8, env.n_agents, env.observation_dim) == (8, 4, 16)
    actions = jax.random.uniform(jax.random.PRNGKey(1), (8, 4, 2), minval=-1, maxval=1)
    obs, state, reward, terminated, truncated, info = jax.jit(jax.vmap(env.step))(
        state, actions
    )
    assert obs.shape == (8, 4, 16)
    assert terminated.shape == truncated.shape == (8,)
    assert reward.shape == info["shared_reward"].shape == (8, 4)
    assert info["difference_rewards"].shape == (8, 4)
    assert info["task_reward"].shape == (8,)
    assert np.all(np.isfinite(np.asarray(obs)))
    assert np.all(np.asarray(state.t) == 1)
    assert (env.action_dim, env.discrete) == (2, False)


def test_reset_spawns_inside_the_spawn_region():
    env = make_env(n_agents=64, spawn_region=(4.0, 6.0, 1.0, 2.0))
    _, state = env.reset(jax.random.PRNGKey(3))
    pos, heading = np.asarray(state.rovers.position), np.asarray(state.rovers.heading)
    assert np.all((pos[:, 0] >= 4.0) & (pos[:, 0] <= 6.0))
    assert np.all((pos[:, 1] >= 1.0) & (pos[:, 1] <= 2.0))
    assert np.all((heading >= -math.pi) & (heading < math.pi))
    np.testing.assert_array_equal(np.asarray(state.pois.remaining), [1.0])


# -------------------------------------------------------------- observation


def test_sectors_are_heading_relative():
    env = make_env()  # R = arena diagonal
    R = env.world_diagonal
    ahead = env.observe(state_at(env, [[5, 5], [7, 5]]))
    np.testing.assert_allclose(ahead[0, :8], np.eye(8)[0] * (1 - 2 / R), rtol=1e-6)
    # the teammate sees rover 0 directly behind it
    np.testing.assert_allclose(ahead[1, :8], np.eye(8)[4] * (1 - 2 / R), rtol=1e-6)
    # +90 degrees in the world with heading 0 is the rover's left: sector 2
    left = env.observe(state_at(env, [[5, 5], [5, 7]]))
    assert int(np.argmax(left[0, :8])) == 2
    # turning the rover +90 degrees puts the teammate ahead on its right: sector 6
    turned = env.observe(state_at(env, [[5, 5], [7, 5]], headings=[math.pi / 2, 0.0]))
    assert int(np.argmax(turned[0, :8])) == 6


def test_sector_zero_spans_plus_minus_half_a_sector():
    env = make_env()
    for degrees, sector in [(22.0, 0), (23.0, 1), (-22.0, 0), (-23.0, 7)]:
        a = math.radians(degrees)
        obs = env.observe(state_at(env, [[5, 5], [5 + 2 * math.cos(a), 5 + 2 * math.sin(a)]]))
        assert int(np.argmax(obs[0, :8])) == sector, degrees


def test_presence_is_a_distance_weighted_share_within_the_radius():
    env = make_env(n_agents=4, observation=SectorDensityObservation(sensor_radius=3.0))
    # teammates ahead at 1.0 and 2.5; the one at 3.5 is out of range
    obs = env.observe(state_at(env, [[5, 5], [6, 5], [7.5, 5], [8.5, 5]]))
    expected = ((1 - 1 / 3) + (1 - 2.5 / 3)) / 3
    np.testing.assert_allclose(obs[0, 0], expected, rtol=1e-6)
    np.testing.assert_allclose(obs[0, 1:8], 0.0)


def test_co_located_teammates_are_sensed_and_presence_reaches_one():
    env = make_env(n_agents=3)
    obs = env.observe(state_at(env, [[5, 5], [5, 5], [5, 5]]))
    np.testing.assert_allclose(obs[:, 0], 1.0)  # self excluded by index, not distance
    assert float(obs.max()) <= 1.0


def test_used_up_pois_are_sensed_like_live_ones_unless_flagged():
    entry = {"position": [7.0, 5.0], "coupling": 2}
    env = make_env(entries=[entry])
    R = env.world_diagonal
    live = env.observe(state_at(env, [[5, 5], [1, 1]], remaining=[0.4]))
    np.testing.assert_allclose(live[0, 8:], np.eye(8)[0] * (1 - 2 / R), rtol=1e-6)
    used = env.observe(state_at(env, [[5, 5], [1, 1]], remaining=[0.0]))
    np.testing.assert_array_equal(used, live)
    # hide_when_used_up: sensed while live, gone once used up (the old rule).
    flagged = make_env(entries=[{**entry, "hide_when_used_up": True}])
    np.testing.assert_array_equal(
        flagged.observe(state_at(flagged, [[5, 5], [1, 1]], remaining=[0.4])), live
    )
    hidden = flagged.observe(state_at(flagged, [[5, 5], [1, 1]], remaining=[0.0]))
    np.testing.assert_allclose(hidden[:, 8:], 0.0)


def test_observations_stay_in_unit_interval_over_random_rollouts():
    for variant in list_variants():
        env = load_rover_env(variant)

        def episode(key):
            obs, state = env.reset(key)

            def body(carry, k):
                state, lo, hi = carry
                a = jax.random.uniform(k, (env.n_agents, 2), minval=-1, maxval=1)
                obs, state, *_ = env.step(state, a)
                return (state, jnp.minimum(lo, obs.min()), jnp.maximum(hi, obs.max())), None

            (_, lo, hi), _ = jax.lax.scan(
                body, (state, obs.min(), obs.max()), jax.random.split(key, env.max_steps)
            )
            return lo, hi

        lo, hi = jax.jit(jax.vmap(episode))(jax.random.split(jax.random.PRNGKey(0), 16))
        assert float(lo.min()) >= 0.0 and float(hi.max()) <= 1.0, variant


# ----------------------------------------------------------------- dynamics


def test_turn_then_move_at_the_set_speed():
    env = make_env(max_speed=0.5, max_turn=math.pi / 4)
    state = state_at(env, [[5, 5], [2, 2]])
    _, nxt, *_ = env.step(state, jnp.array([[1.0, 1.0], [0.0, -1.0]]))
    np.testing.assert_allclose(nxt.rovers.heading[0], math.pi / 4, atol=1e-6)
    step = 0.5 * math.sqrt(0.5)
    np.testing.assert_allclose(nxt.rovers.position[0], [5 + step, 5 + step], atol=1e-6)
    np.testing.assert_allclose(nxt.rovers.position[1], [2, 2])  # speed action -1 = stop
    # half speed, turning right by half the maximum
    _, half, *_ = env.step(state, jnp.array([[-0.5, 0.0], [0.0, -1.0]]))
    np.testing.assert_allclose(half.rovers.heading[0], -math.pi / 8, atol=1e-6)
    np.testing.assert_allclose(
        half.rovers.position[0],
        [5 + 0.25 * math.cos(math.pi / 8), 5 - 0.25 * math.sin(math.pi / 8)],
        atol=1e-6,
    )
    # out-of-range actions are clipped to [-1, 1]
    _, clipped, *_ = env.step(state, jnp.array([[5.0, 9.0], [0.0, -3.0]]))
    np.testing.assert_allclose(clipped.rovers.position, nxt.rovers.position)


def test_heading_wraps_and_walls_clip():
    env = make_env(max_speed=0.5, max_turn=math.pi / 4)
    state = state_at(env, [[9.9, 5.0], [5.0, 0.1]], headings=[3.0, -math.pi / 2])
    _, nxt, _, terminated, *_ = env.step(state, jnp.array([[1.0, -1.0], [0.0, 1.0]]))
    expected = (3.0 + math.pi / 4 + math.pi) % (2 * math.pi) - math.pi
    np.testing.assert_allclose(nxt.rovers.heading[0], expected, atol=1e-6)
    np.testing.assert_allclose(nxt.rovers.position[1], [5.0, 0.0], atol=1e-6)
    _, wall, _, terminated, *_ = env.step(
        state_at(env, [[9.9, 5.0], [5.0, 5.0]]), jnp.array([[0.0, 1.0], [0.0, -1.0]])
    )
    np.testing.assert_allclose(wall.rovers.position[0], [10.0, 5.0])
    assert not bool(terminated)  # walls never end the episode


# ------------------------------------------------------------------- reward


def test_reward_needs_the_coupling_and_is_shared_by_every_rover():
    env = make_env(n_agents=3)
    _, _, r, _, _, info = env.step(state_at(env, [[5, 5], [1, 1], [9, 9]]), stay(3))
    assert float(info["task_reward"]) == 0.0 and float(info["poi_remaining"][0]) == 1.0
    np.testing.assert_array_equal(r, [0.0, 0.0, 0.0])
    # the radius is inclusive: (6, 5) is exactly 1.0 from the POI
    _, _, r, _, _, info = env.step(state_at(env, [[5, 5], [6, 5], [9, 9]]), stay(3))
    np.testing.assert_allclose(info["task_reward"], 0.05)
    # every rover is paid G, the one far from the POI included
    np.testing.assert_allclose(r, [0.05] * 3)
    np.testing.assert_array_equal(r, info["shared_reward"])
    # a third rover adds nothing to the payout
    _, _, _, _, _, info = env.step(state_at(env, [[5, 5], [6, 5], [4.5, 5]]), stay(3))
    np.testing.assert_allclose(info["task_reward"], 0.05)


def test_difference_rewards_pay_only_rovers_that_complete_a_coupling():
    env = make_env(n_agents=3)
    # alone at a coupling-2 POI: G = 0 with or without it
    _, _, _, _, _, info = env.step(state_at(env, [[5, 5], [1, 1], [9, 9]]), stay(3))
    np.testing.assert_array_equal(info["difference_rewards"], [0.0, 0.0, 0.0])
    # exactly the coupling: each observer is needed, so each earns the payout
    _, _, _, _, _, info = env.step(state_at(env, [[5, 5], [6, 5], [9, 9]]), stay(3))
    np.testing.assert_allclose(info["difference_rewards"], [0.05, 0.05, 0.0])
    # one beyond the coupling: the POI still pays without any single rover
    _, _, _, _, _, info = env.step(state_at(env, [[5, 5], [6, 5], [4.5, 5]]), stay(3))
    np.testing.assert_allclose(info["task_reward"], 0.05)
    np.testing.assert_array_equal(info["difference_rewards"], [0.0, 0.0, 0.0])
    # a used-up POI pays nothing either way
    _, _, _, _, _, info = env.step(
        state_at(env, [[5, 5], [6, 5], [9, 9]], remaining=[0.0]), stay(3)
    )
    np.testing.assert_array_equal(info["difference_rewards"], [0.0, 0.0, 0.0])
    # rover 0 sits in both radii and completes both couplings; rover 1 only B's
    env = make_env(entries=[{"position": [5.0, 5.0], "coupling": 1},
                            {"position": [6.5, 5.0], "coupling": 2}])
    _, _, _, _, _, info = env.step(state_at(env, [[5.75, 5], [7, 5]]), stay(2))
    np.testing.assert_allclose(info["task_reward"], 0.1)
    np.testing.assert_allclose(info["difference_rewards"], [0.1, 0.05])


@pytest.mark.parametrize("orders", [None, (0, 1, 1, 2)])
def test_difference_rewards_match_the_team_reward_without_each_rover(orders):
    """D_i = G - G_{-i}, with G_{-i} from the reward model run on the other rovers.

    With ``orders`` the random budgets also lock and unlock POIs.
    """
    cfg = load_env_config("rover_4poi_c1234")
    for entry, order in zip(cfg["pois"]["entries"], orders or ()):
        entry["order"] = order
    env = make_rover_env(cfg)
    _, state = env.reset(jax.random.PRNGKey(0))
    @jax.jit
    def reward(rovers, pois):
        out = env.reward(env, rovers, pois)
        return out.team_reward, out.info["difference_rewards"]

    n_agents, n_pois = env.n_agents, env.n_pois
    rng = np.random.default_rng(0)
    n_with_credit = 0
    for _ in range(64):
        # every rover near a random POI, some just outside its radius
        near = np.asarray(state.pois.position)[rng.integers(0, n_pois, n_agents)]
        rovers = RoverState(
            jnp.asarray(near + rng.uniform(-0.9, 0.9, (n_agents, 2)), jnp.float32),
            jnp.zeros(n_agents),
        )
        remaining = np.where(rng.random(n_pois) < 0.25, 0.0, 1.0)
        pois = dataclasses.replace(state.pois, remaining=jnp.asarray(remaining, jnp.float32))
        team, difference = reward(rovers, pois)
        for i in range(n_agents):
            keep = np.arange(n_agents) != i
            team_without, _ = reward(
                RoverState(rovers.position[keep], rovers.heading[keep]), pois
            )
            np.testing.assert_allclose(difference[i], team - team_without, atol=1e-6)
        n_with_credit += int(np.any(np.asarray(difference) > 0))
    assert n_with_credit > 10  # the draws exercise the nonzero case


@pytest.mark.parametrize("mode, per_agent", [
    ("shared_reward", False), ("difference_rewards", True),
])
def test_the_mode_picks_the_per_rover_reward_and_the_team_reward_stays_g(mode, per_agent):
    env = make_env(n_agents=3, reward=CouplingDepletionReward(mode))
    assert env.per_agent_rewards is per_agent and env.reward_mode == mode
    # one rover beyond the coupling: every D_i is 0 while G is 0.05
    _, _, r, _, _, info = env.step(state_at(env, [[5, 5], [6, 5], [4.5, 5]]), stay(3))
    np.testing.assert_array_equal(r, info[mode])
    np.testing.assert_allclose(info["task_reward"], 0.05)


def test_rollouts_score_the_team_reward_under_difference_rewards():
    env = load_rover_env("rover_2poi_c2", ["env.reward.mode=difference_rewards"])
    traj = rollout(env, make_scripted_coalition_policy(env), jax.random.PRNGKey(0))
    # two rovers complete each coupling, so the D_i sum to 2 G; the return is G's
    np.testing.assert_allclose(traj.episode_return, 2.0, atol=1e-5)


def test_poi_pays_a_steady_rate_until_used_up():
    env = make_env(max_steps=100)
    state = state_at(env, [[5, 5], [5.5, 5]])
    rewards, terminated = [], []
    for _ in range(25):
        _, state, _, term, trunc, info = env.step(state, stay(2))
        rewards.append(float(info["task_reward"]))
        terminated.append(bool(term))
        assert not (bool(term) and bool(trunc))
    np.testing.assert_allclose(rewards[:20], 0.05, rtol=1e-5)
    assert rewards[20:] == [0.0] * 5
    np.testing.assert_allclose(sum(rewards), 1.0, atol=1e-5)
    assert terminated.index(True) == 19  # exactly value / rate coupled steps
    assert float(state.pois.remaining[0]) == 0.0


def test_termination_needs_every_poi_used_up():
    env = make_env(entries=[{"position": [5.0, 5.0], "coupling": 2},
                            {"position": [1.0, 1.0], "coupling": 2}])
    _, _, _, terminated, *_ = env.step(
        state_at(env, [[5, 5], [5, 5]], remaining=[0.05, 1.0]), stay(2)
    )
    assert not bool(terminated)
    _, _, _, terminated, *_ = env.step(
        state_at(env, [[5, 5], [5, 5]], remaining=[0.05, 0.0]), stay(2)
    )
    assert bool(terminated)


def test_truncation_at_max_steps_is_exclusive_with_termination():
    env = make_env(max_steps=5)
    state = state_at(env, [[1, 1], [9, 9]])
    flags = []
    for _ in range(5):
        _, state, _, term, trunc, _ = env.step(state, stay(2))
        flags.append((bool(term), bool(trunc)))
    assert flags == [(False, False)] * 4 + [(False, True)]
    # the last budgeted step ends both ways at once: report termination only
    env = make_env(max_steps=1)
    _, _, _, term, trunc, _ = env.step(state_at(env, [[5, 5], [5, 5]], remaining=[0.05]), stay(2))
    assert (bool(term), bool(trunc)) == (True, False)


# -------------------------------------------------------- temporal coupling


def test_orders_default_to_zero_and_leave_every_poi_unlocked():
    env = load_rover_env("rover_4poi_c1234")
    pois = env.pois.init(jax.random.PRNGKey(0), env)
    np.testing.assert_array_equal(np.asarray(pois.order), [0, 0, 0, 0])
    assert pois.order.dtype == jnp.int32
    assert np.all(np.asarray(unlocked(pois)))


def test_a_poi_unlocks_once_every_lower_order_is_used_up():
    """Ties share a stage, gaps in the numbering do not matter, and only lower
    orders lock a POI (its own budget does not)."""
    env = make_env(entries=[{"position": [2.0, 2.0], "coupling": 1, "order": 0},
                            {"position": [8.0, 2.0], "coupling": 1, "order": 2},
                            {"position": [8.0, 8.0], "coupling": 1, "order": 2},
                            {"position": [2.0, 8.0], "coupling": 1, "order": 5}])
    pois = env.pois.init(jax.random.PRNGKey(0), env)
    remaining = jnp.asarray([[1.0, 1.0, 1.0, 1.0],
                             [0.0, 1.0, 1.0, 1.0],
                             [0.0, 0.0, 1.0, 1.0],
                             [0.0, 1.0, 0.0, 1.0],
                             [0.0, 0.0, 0.0, 1.0],
                             [1.0, 0.0, 0.0, 0.0]])
    batched = jax.vmap(lambda r: unlocked(dataclasses.replace(pois, remaining=r)))(remaining)
    np.testing.assert_array_equal(np.asarray(batched), [[True, False, False, False],
                                                        [True, True, True, False],
                                                        [True, True, True, False],
                                                        [True, True, True, False],
                                                        [True, True, True, True],
                                                        [True, False, False, False]])


def test_a_later_poi_pays_only_from_the_step_after_the_earlier_one_is_used_up():
    env = make_env(entries=[{"position": [2.0, 5.0], "coupling": 1, "order": 0},
                            {"position": [8.0, 5.0], "coupling": 1, "order": 1}])
    state = state_at(env, [[2, 5], [8, 5]])  # each coupling met throughout
    payouts, unlocks, terminated = [], [], []
    for _ in range(45):
        _, state, _, term, _, info = env.step(state, stay(2))
        payouts.append(np.asarray(info["poi_payout"]))
        unlocks.append(np.asarray(info["poi_unlocked"]))
        terminated.append(bool(term))
    payouts, unlocks = np.stack(payouts), np.stack(unlocks)
    np.testing.assert_allclose(payouts[:20, 0], 0.05, rtol=1e-5)
    assert np.all(payouts[:20, 1] == 0.0) and not unlocks[:20, 1].any()
    # P0's last payout is at step 19; P1 is unlocked from step 20 on
    assert unlocks[20:, 1].all() and np.all(payouts[20:, 0] == 0.0)
    np.testing.assert_allclose(payouts[20:40, 1], 0.05, rtol=1e-5)
    np.testing.assert_allclose(payouts.sum(), 2.0, atol=1e-5)
    assert terminated.index(True) == 39


def test_a_locked_poi_pays_no_one_even_with_its_coupling_met():
    env = make_env(n_agents=3, entries=[{"position": [2.0, 5.0], "coupling": 1, "order": 0},
                                        {"position": [8.0, 5.0], "coupling": 2, "order": 1}])
    _, _, r, _, _, info = env.step(state_at(env, [[8, 5], [8.5, 5], [5, 5]]), stay(3))
    np.testing.assert_array_equal(np.asarray(info["poi_active"]), [False, False])
    assert float(info["task_reward"]) == 0.0
    np.testing.assert_array_equal(r, [0.0, 0.0, 0.0])
    np.testing.assert_array_equal(info["difference_rewards"], [0.0, 0.0, 0.0])
    # once P0 is used up the same positions complete P1's coupling
    _, _, _, _, _, info = env.step(
        state_at(env, [[8, 5], [8.5, 5], [5, 5]], remaining=[0.0, 1.0]), stay(3)
    )
    np.testing.assert_allclose(info["difference_rewards"], [0.05, 0.05, 0.0])


def test_the_scripted_team_works_ordered_pois_in_their_order():
    """Two rovers, P1 ordered before P0: index order alone would park both at
    the locked P0 forever."""
    cfg = load_env_config("rover_4a_2poi_c2", ["env.n_agents=2"])
    cfg["pois"]["entries"][0]["order"] = 1
    env = make_rover_env(cfg)
    traj = rollout(env, make_scripted_coalition_policy(env), jax.random.PRNGKey(0))
    np.testing.assert_allclose(traj.episode_return, 2.0, atol=1e-5)
    remaining = np.asarray(traj.poi_remaining)
    used_at = [int(np.argmax(remaining[:, k] <= 0)) for k in range(2)]
    assert used_at[1] < used_at[0]
    # Not enforced, the order locks nothing: the team goes back to index order.
    cfg["pois"]["enforce_order"] = False
    with pytest.warns(UserWarning, match="the order has no effect"):
        env = make_rover_env(cfg)
    traj = rollout(env, make_scripted_coalition_policy(env), jax.random.PRNGKey(0))
    np.testing.assert_allclose(traj.episode_return, 2.0, atol=1e-5)
    remaining = np.asarray(traj.poi_remaining)
    used_at = [int(np.argmax(remaining[:, k] <= 0)) for k in range(2)]
    assert used_at[0] < used_at[1]


def test_hide_until_unlocked_defaults_off_and_an_entry_overrides_the_block_default():
    entries = [{"position": [2.0, 5.0], "coupling": 1, "order": 0},
               {"position": [8.0, 5.0], "coupling": 1, "order": 1, "hide_until_unlocked": True},
               {"position": [5.0, 8.0], "coupling": 1, "order": 1}]
    _, state = make_env(entries=entries).reset(jax.random.PRNGKey(0))
    np.testing.assert_array_equal(state.pois.hide_until_unlocked, [False, True, False])
    assert state.pois.hide_until_unlocked.dtype == jnp.bool_
    entries[0]["hide_until_unlocked"] = entries[1]["hide_until_unlocked"] = False
    pois = StaticPOIs(entries, hide_until_unlocked=True)
    np.testing.assert_array_equal(pois.hide_until_unlocked, [False, False, True])


def test_a_flagged_poi_is_hidden_until_it_unlocks():
    """P0 behind the rover (sector 4), P1 ahead (sector 0) and ordered after P0."""
    entries = [{"position": [3.0, 5.0], "coupling": 1, "order": 0},
               {"position": [7.0, 5.0], "coupling": 1, "order": 1}]
    plain = make_env(entries=entries)
    entries[1]["hide_until_unlocked"] = True
    env = make_env(entries=entries)
    one = (1 - 2 / env.world_diagonal) / 2  # one POI at distance 2, of 2 POIs

    locked = env.observe(state_at(env, [[5, 5], [1, 1]], remaining=[1.0, 1.0]))
    np.testing.assert_allclose(locked[0, 8:], np.eye(8)[4] * one, rtol=1e-6)
    unflagged = plain.observe(state_at(plain, [[5, 5], [1, 1]], remaining=[1.0, 1.0]))
    np.testing.assert_allclose(unflagged[0, 8:], (np.eye(8)[0] + np.eye(8)[4]) * one, rtol=1e-6)
    # Unlocked: sensed with the same presence as an unflagged POI (same
    # normalizer), and the used-up P0 is still sensed behind the rover.
    unlocked_obs = env.observe(state_at(env, [[5, 5], [1, 1]], remaining=[0.0, 1.0]))
    np.testing.assert_allclose(
        unlocked_obs[0, 8:], (np.eye(8)[0] + np.eye(8)[4]) * one, rtol=1e-6
    )


def observe_along(env, traj, remaining, key=jax.random.PRNGKey(0)):
    """(T+1, A, obs_dim) observations of ``env`` at a trajectory's poses, with
    the POI budgets ``remaining`` (T+1, P) in place of the env's own."""
    pois = env.reset(key)[1].pois
    return np.asarray(jax.vmap(lambda p, h, r: env.observe(RoverEnvState(
        RoverState(p, h), dataclasses.replace(pois, remaining=r), jnp.zeros((), jnp.int32)
    )))(traj.positions, traj.headings, remaining))


def scripted_episode(env, key=jax.random.PRNGKey(0)):
    return jax.jit(lambda k: rollout(env, make_scripted_coalition_policy(env), k))(key)


def test_a_flagged_poi_is_absent_while_locked_and_the_task_is_unchanged():
    """A jitted episode of rover_4a_2poi_c2_ord with P1 flagged: while P0 has
    budget the observation equals one without P1 (P1 used up under
    hide_when_used_up, same normalizer); once P0 is used up it equals the
    unflagged env's. Trajectories and rewards are identical."""
    def with_p1(**flags):
        cfg = load_env_config("rover_4a_2poi_c2_ord")
        cfg["pois"]["entries"][1].update(flags)
        return make_rover_env(cfg)

    plain = with_p1()
    without_p1 = with_p1(hide_when_used_up=True)
    env = with_p1(hide_until_unlocked=True)
    traj, plain_traj = scripted_episode(env), scripted_episode(plain)
    np.testing.assert_array_equal(traj.positions, plain_traj.positions)
    np.testing.assert_array_equal(traj.rewards, plain_traj.rewards)

    remaining = traj.poi_remaining
    obs = observe_along(env, traj, remaining)
    absent = observe_along(without_p1, traj, remaining.at[:, 1].set(0.0))
    unflagged = observe_along(plain, traj, remaining)
    locked = np.asarray(remaining[:, 0] > 0)
    after = ~locked & np.asarray(remaining[:, 1] > 0)
    assert locked.any() and after.any()  # the episode reaches both phases
    np.testing.assert_array_equal(obs[locked], absent[locked])
    np.testing.assert_array_equal(obs[after], unflagged[after])
    assert not np.array_equal(obs[locked], unflagged[locked])


def test_hide_when_used_up_defaults_off_in_every_variant_and_an_entry_overrides_it():
    entries = [{"position": [2.0, 5.0], "coupling": 1},
               {"position": [8.0, 5.0], "coupling": 1, "hide_when_used_up": True}]
    _, state = make_env(entries=entries).reset(jax.random.PRNGKey(0))
    np.testing.assert_array_equal(state.pois.hide_when_used_up, [False, True])
    assert state.pois.hide_when_used_up.dtype == jnp.bool_
    entries[1]["hide_when_used_up"] = False
    pois = StaticPOIs(entries, hide_when_used_up=True)
    np.testing.assert_array_equal(pois.hide_when_used_up, [True, False])
    with pytest.raises(ValueError, match="hide_when_used_up must be true or false"):
        make_env(entries=[{"position": [5.0, 5.0], "coupling": 1, "hide_when_used_up": 1}])
    for variant in list_variants():
        _, state = load_rover_env(variant).reset(jax.random.PRNGKey(0))
        assert not state.pois.hide_when_used_up.any(), variant


def test_used_up_pois_change_the_observation_only_once_used_up():
    """Over a scripted episode of rover_4a_2poi_c2_ord, the default and the old
    rule (hide_when_used_up) observe the same until a POI is used up, and differ
    afterwards wherever a rover is in range of the used-up POI."""
    env = load_rover_env("rover_4a_2poi_c2_ord")
    old_rule = load_rover_env("rover_4a_2poi_c2_ord", ["env.pois.hide_when_used_up=true"])
    traj = scripted_episode(env)
    remaining = traj.poi_remaining
    new = observe_along(env, traj, remaining)
    old = observe_along(old_rule, traj, remaining)
    none_used = np.asarray((remaining > 0).all(axis=1))
    assert none_used.any() and (~none_used).any()
    np.testing.assert_array_equal(new[none_used], old[none_used])
    # Global sensing: once a POI is used up, every rover still senses it by default.
    assert (new[~none_used, :, 8:] > old[~none_used, :, 8:]).any(axis=(1, 2)).all()
    np.testing.assert_array_equal(new[..., :8], old[..., :8])  # agent block untouched


def test_hide_until_unlocked_must_be_a_bool_and_warns_where_it_never_acts():
    with pytest.raises(ValueError, match="hide_until_unlocked must be true or false"):
        make_env(entries=[{"position": [5.0, 5.0], "coupling": 1, "hide_until_unlocked": 1}])
    entries = [{"position": [2.0, 5.0], "coupling": 1, "order": 0},
               {"position": [8.0, 5.0], "coupling": 1, "order": 1}]
    with pytest.warns(UserWarning, match=r"POIs \[0\], but none of them"):
        make_env(entries=[{**entries[0], "hide_until_unlocked": True}, entries[1]])
    with pytest.warns(UserWarning, match=r"POIs \[0, 1\], but none of them"):
        StaticPOIs([{**e, "order": 0} for e in entries], hide_until_unlocked=True)
    # Flagging every POI of an ordered layout acts on P1, so it is no mistake.
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        StaticPOIs(entries, hide_until_unlocked=True)


def test_enforce_order_defaults_on_and_an_entry_overrides_the_block_default():
    entries = [{"position": [2.0, 5.0], "coupling": 1, "order": 0},
               {"position": [8.0, 5.0], "coupling": 1, "order": 1, "enforce_order": False},
               {"position": [5.0, 8.0], "coupling": 1, "order": 1}]
    _, state = make_env(entries=entries).reset(jax.random.PRNGKey(0))
    np.testing.assert_array_equal(state.pois.enforce_order, [True, False, True])
    assert state.pois.enforce_order.dtype == jnp.bool_
    entries[1]["enforce_order"] = True
    pois = StaticPOIs(entries, enforce_order=False)
    np.testing.assert_array_equal(pois.enforce_order, [False, True, False])
    with pytest.raises(ValueError, match="enforce_order must be true or false"):
        make_env(entries=[{"position": [5.0, 5.0], "coupling": 1, "enforce_order": 0}])


def test_an_order_that_is_not_enforced_locks_no_reward():
    """The layout of test_a_locked_poi_pays_no_one_even_with_its_coupling_met,
    with P1's order not enforced: P1 pays (difference rewards included) while
    P0 still has budget, though its order is not reached."""
    with pytest.warns(UserWarning, match="the order has no effect"):
        env = make_env(n_agents=3, entries=[
            {"position": [2.0, 5.0], "coupling": 1, "order": 0},
            {"position": [8.0, 5.0], "coupling": 2, "order": 1, "enforce_order": False}])
    state = state_at(env, [[8, 5], [8.5, 5], [5, 5]])
    np.testing.assert_array_equal(np.asarray(order_reached(state.pois)), [True, False])
    _, _, r, _, _, info = env.step(state, stay(3))
    np.testing.assert_array_equal(np.asarray(info["poi_unlocked"]), [True, True])
    np.testing.assert_array_equal(np.asarray(info["poi_active"]), [False, True])
    np.testing.assert_allclose(float(info["task_reward"]), 0.05, rtol=1e-6)
    np.testing.assert_allclose(r, [0.05, 0.05, 0.05], rtol=1e-6)
    np.testing.assert_allclose(info["difference_rewards"], [0.05, 0.05, 0.0], rtol=1e-6)


def test_an_order_that_is_not_enforced_and_hides_nothing_is_the_unordered_env():
    """rover_4a_2poi_c2_ord with enforce_order false: scripted episodes,
    rewards and observations equal rover_4a_2poi_c2's."""
    with pytest.warns(UserWarning, match="the order has no effect"):
        env = load_rover_env("rover_4a_2poi_c2_ord", ["env.pois.enforce_order=false"])
    plain = load_rover_env("rover_4a_2poi_c2")
    for key in jax.random.split(jax.random.PRNGKey(3), 3):
        traj, plain_traj = scripted_episode(env, key), scripted_episode(plain, key)
        np.testing.assert_array_equal(traj.positions, plain_traj.positions)
        np.testing.assert_array_equal(traj.rewards, plain_traj.rewards)
        np.testing.assert_array_equal(
            observe_along(env, traj, traj.poi_remaining, key),
            observe_along(plain, traj, traj.poi_remaining, key),
        )


def test_hide_until_unlocked_follows_the_order_when_it_is_not_enforced():
    """The layout of test_a_flagged_poi_is_hidden_until_it_unlocks, P1 flagged:
    enforced or not, the rovers sense the same at every budget, P1 appearing
    once P0 is used up. Not enforced, P1 pays (and can be used up) while still
    hidden."""
    entries = [{"position": [3.0, 5.0], "coupling": 1, "order": 0},
               {"position": [7.0, 5.0], "coupling": 1, "order": 1, "hide_until_unlocked": True}]
    enforced = make_env(entries=entries)
    entries[1]["enforce_order"] = False
    free = make_env(entries=entries)
    one = (1 - 2 / free.world_diagonal) / 2  # one POI at distance 2, of 2 POIs
    for remaining in ([1.0, 1.0], [0.0, 1.0], [1.0, 0.0], [0.0, 0.0]):
        obs = free.observe(state_at(free, [[5, 5], [1, 1]], remaining=remaining))
        np.testing.assert_array_equal(
            obs, enforced.observe(state_at(enforced, [[5, 5], [1, 1]], remaining=remaining))
        )
        p1_sensed = remaining[0] == 0.0  # P1 appears once P0 is used up
        np.testing.assert_allclose(
            obs[0, 8:], (np.eye(8)[4] + p1_sensed * np.eye(8)[0]) * one, rtol=1e-6
        )
    # Rover 0 on the hidden P1 while P0 has budget: paid only when not enforced.
    for env, paid in ((enforced, 0.0), (free, 0.05)):
        _, _, _, _, _, info = env.step(state_at(env, [[7, 5], [1, 1]]), stay(2))
        np.testing.assert_allclose(np.asarray(info["poi_payout"]), [0.0, paid], rtol=1e-6)


def test_enforce_order_warns_where_it_changes_nothing():
    entries = [{"position": [2.0, 5.0], "coupling": 1, "order": 0},
               {"position": [8.0, 5.0], "coupling": 1, "order": 1}]
    # Unenforced on the lowest order only: no POI it could act on.
    with pytest.warns(UserWarning, match=r"enforce_order is false on POIs \[0\]"):
        make_env(entries=[{**entries[0], "enforce_order": False}, entries[1]])
    # Unenforced on an unordered layout: there is no order to enforce.
    with pytest.warns(UserWarning, match=r"enforce_order is false on POIs \[0, 1\]"):
        StaticPOIs([{**e, "order": 0} for e in entries], enforce_order=False)
    # Unenforced on every POI and nothing hidden: the order acts on nothing.
    with pytest.warns(UserWarning, match=r"POIs \[1\] have an order above the lowest"):
        StaticPOIs(entries, enforce_order=False)
    # Unenforced but revealed in order: the intended use, no warning.
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        StaticPOIs(entries, enforce_order=False, hide_until_unlocked=True)


# ------------------------------------------------------------ pluggability


class DriftingPOIs(POIModel):
    """One POI that moves +0.5 in x every step."""

    def init(self, key, env):
        return StaticPOIs([{"position": [2.0, 5.0], "coupling": 1}]).init(key, env)

    def step(self, pois, rovers, t):
        return dataclasses.replace(pois, position=pois.position + jnp.array([0.5, 0.0]))


class OwnPosition(ObservationModel):
    obs_dim = 2

    def __call__(self, env, state):
        return state.rovers.position / jnp.asarray(env.world_size)


class NegativeDistance(RewardModel):
    """Per-agent reward: minus the distance to the first POI; never terminates."""

    def __call__(self, env, rovers, pois):
        d = jnp.linalg.norm(rovers.position - pois.position[0], axis=-1)
        return RewardOutput(-d, -d.sum(), pois, jnp.array(False), {"distance": d})


def test_components_are_pluggable():
    env = BaseRoverEnv(2, DriftingPOIs(), OwnPosition(), NegativeDistance())
    assert env.observation_dim == 2 and env.n_pois == 1
    obs, state = env.reset(jax.random.PRNGKey(0))
    np.testing.assert_allclose(obs, state.rovers.position / 10.0)
    _, state2, r, terminated, _, info = env.step(state, stay(2))
    np.testing.assert_allclose(state2.pois.position[0], [2.5, 5.0])
    assert r.shape == (2,)
    np.testing.assert_allclose(r, -info["distance"])
    np.testing.assert_allclose(info["task_reward"], r.sum())
    assert not bool(terminated)


def test_an_observation_model_with_the_wrong_declared_width_is_rejected():
    class Wrong(OwnPosition):
        obs_dim = 3

    with pytest.raises(ValueError, match="obs_dim"):
        BaseRoverEnv(2, DriftingPOIs(), Wrong())


def test_a_reward_model_without_one_reward_per_rover_is_rejected():
    class TeamOnly(NegativeDistance):
        def __call__(self, env, rovers, pois):
            out = super().__call__(env, rovers, pois)
            return dataclasses.replace(out, reward=out.team_reward)

    with pytest.raises(ValueError, match="must return reward of shape"):
        BaseRoverEnv(2, DriftingPOIs(), OwnPosition(), TeamOnly())


# ------------------------------------------------------------------- config


def test_every_rover_variant_composes_and_builds():
    variants = list_variants()
    assert "rover_4a_2poi_c2" in variants and len(variants) >= 3
    default = load_env_config("rover_4a_2poi_c2")
    for variant in variants:
        cfg = load_env_config(variant)
        env = make_rover_env(cfg)
        assert env.observation_dim == 16
        # Children override the POIs, the team size, the arena and the sensor
        # radius; the dynamics and the rest of the observation are inherited.
        assert cfg["dynamics"] == default["dynamics"]
        observation = {k: v for k, v in cfg["observation"].items() if k != "sensor_radius"}
        default_observation = {
            k: v for k, v in default["observation"].items() if k != "sensor_radius"
        }
        assert observation == default_observation


def test_default_variant_is_two_pois_of_coupling_two():
    env = load_rover_env("rover_2poi_c2")
    pois = env.pois.init(jax.random.PRNGKey(0), env)
    assert env.n_pois == 2
    np.testing.assert_array_equal(np.asarray(pois.coupling), [2, 2])


def test_hydra_overrides_reach_the_env():
    env = load_rover_env("rover_2poi_c2", overrides=["env.n_agents=6", "env.max_steps=40"])
    assert (env.n_agents, env.max_steps) == (6, 40)


@pytest.mark.parametrize(
    "edit, message",
    [
        (lambda c: c.update(bogus=1), "unknown rover env keys"),
        (lambda c: c["dynamics"].update(max_sped=1.0), "unknown dynamics keys"),
        (lambda c: c["observation"].update(type="lidar"), "unknown observation type"),
        (lambda c: c["reward"].update(mode="team"), "unknown reward mode"),
        (lambda c: c.update(environment="multi_box_push"), "environment must be"),
        (lambda c: c["pois"]["entries"][0].update(cupling=2), "unknown keys"),
        (lambda c: c["pois"]["entries"][0].update(position=[11.0, 5.0]), "outside"),
        (lambda c: c["pois"]["entries"][0].update(coupling=0), "coupling"),
        (lambda c: c["pois"]["entries"][0].update(order=-1), "order"),
        (lambda c: c["pois"]["entries"][0].update(order=1.5), "order"),
        (lambda c: c.update(spawn_region=[4.0, 12.0, 4.0, 6.0]), "spawn_region"),
    ],
)
def test_bad_config_raises(edit, message):
    cfg = load_env_config("rover_2poi_c2")
    edit(cfg)
    with pytest.raises(ValueError, match=message):
        make_rover_env(cfg)


def test_a_poi_needing_more_rovers_than_exist_warns():
    with pytest.warns(UserWarning, match="can never pay"):
        make_env(n_agents=2, entries=[{"position": [5.0, 5.0], "coupling": 3}])


def test_pois_ordered_after_one_that_can_never_pay_warn():
    entries = [{"position": [5.0, 5.0], "coupling": 3, "order": 1},
               {"position": [2.0, 2.0], "coupling": 1, "order": 2},
               {"position": [8.0, 8.0], "coupling": 1, "order": 0}]
    with pytest.warns(UserWarning, match="can never pay"), \
            pytest.warns(UserWarning, match=r"POIs \[1\] are ordered after .* never unlock"):
        make_env(n_agents=2, entries=entries)
    # Not enforced, P1 pays anyway; flagged, it is never sensed instead.
    entries[1].update(enforce_order=False, hide_until_unlocked=True)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        make_env(n_agents=2, entries=entries)
    messages = [str(w.message) for w in caught]
    assert any("POIs [1] are ordered after" in m and "never sensed" in m for m in messages)
    assert not any("never unlock" in m for m in messages)


# ----------------------------------------------------------------- rollouts


def test_scripted_team_uses_up_every_poi_and_rollout_freezes_after_the_end():
    env = load_rover_env("rover_2poi_c2")
    policy = make_scripted_coalition_policy(env)
    traj = jax.jit(jax.vmap(lambda k: rollout(env, policy, k)))(
        jax.random.split(jax.random.PRNGKey(0), 4)
    )
    np.testing.assert_allclose(traj.episode_return, 2.0, atol=1e-5)
    done = np.asarray(traj.done)
    assert done[:, -1].all()
    for e in range(4):
        end = int(np.argmax(done[e]))  # step index at which the episode ended
        assert np.all(done[e, end:])
        assert np.all(np.asarray(traj.rewards[e, end + 1:]) == 0.0)
        frozen = np.asarray(traj.positions[e, end + 1:])
        np.testing.assert_array_equal(frozen, np.broadcast_to(frozen[0], frozen.shape))
        np.testing.assert_array_equal(np.asarray(traj.poi_remaining[e, end + 1]), [0.0, 0.0])


def test_rollout_vmaps_over_a_population_of_policy_parameters():
    """Evolutionary-algorithm usage: one fitness per parameter set."""
    env = load_rover_env("rover_2poi_c2")

    def linear_policy(params, key, obs, state):
        return jnp.tanh(obs @ params)

    population = jax.random.normal(jax.random.PRNGKey(0), (5, env.observation_dim, 2))
    keys = jax.random.split(jax.random.PRNGKey(1), 5)
    fitness = jax.jit(jax.vmap(lambda p, k: rollout(env, linear_policy, k, p).episode_return))(
        population, keys
    )
    assert fitness.shape == (5,) and np.all(np.isfinite(np.asarray(fitness)))


ROUTED_VARIANT = "rover_16a_6poi_c844844"


def test_routed_team_solves_its_variant_in_planned_entry_plus_29_steps():
    """The centre POIs pay from the step the last centre-team rover gets
    inside, so they are used up 9 steps later. Then an 11-step leg (disk gap
    5.07 at 0.5 per step) and 10 coupled steps at the corners end the episode
    29 steps after that entry, or up to 2 more when a late rover is still
    turning toward its corner as its centre POI is used up (13% of seeds; at
    the old rate 0.05 the centre took 20 steps and there was always time)."""
    env = load_rover_env(ROUTED_VARIANT)
    keys = jax.vmap(jax.random.PRNGKey)(jnp.arange(8))
    traj, routes = planned_rollouts(env, keys, ROUTE_PLANS[ROUTED_VARIANT])
    np.testing.assert_allclose(traj.episode_return, 6.0, atol=1e-5)
    steps, used_at = solve_steps(traj)
    for i in range(8):
        _, state, _ = rollout_reset(env, keys[i])
        entry = np.asarray(entry_steps(env, state.rovers, state.pois))
        last_entry = max(entry[r, routes[i, r, 0]] for r in range(env.n_agents))
        assert used_at[i, [0, 3]].max() == last_entry + 9
        assert last_entry + 29 <= steps[i] <= last_entry + 31


def assert_routed_rovers_stay_while_needed(env, traj, routes):
    """No routed rover leaves its target's radius while the target has budget."""
    centre, radius = np.asarray(env.pois.position), np.asarray(env.pois.radius)
    pos, remaining = np.asarray(traj.positions), np.asarray(traj.poi_remaining)
    rows = np.arange(env.n_agents)
    for i in range(pos.shape[0]):
        for t in range(env.max_steps):
            left = remaining[i, t][routes[i]] > 0  # (A, K)
            target = routes[i][rows, left.argmax(axis=1)]
            inside = np.linalg.norm(pos[i, t] - centre[target], axis=-1) <= radius[target]
            still_needed = left.any(axis=1) & (remaining[i, t + 1][target] > 0)
            after = np.linalg.norm(pos[i, t + 1] - centre[target], axis=-1) <= radius[target]
            assert np.all(after[inside & still_needed])


def test_routed_rovers_never_leave_a_poi_that_still_needs_them():
    env = load_rover_env(ROUTED_VARIANT)
    keys = jax.vmap(jax.random.PRNGKey)(jnp.arange(4))
    traj, routes = planned_rollouts(env, keys, ROUTE_PLANS[ROUTED_VARIANT])
    assert_routed_rovers_stay_while_needed(env, traj, routes)


# Groups whose POIs need more coupled steps than the rovers can supply in one
# episode: a floor below their plan's return over 1000 seeds (range in the
# comment). A `<name>_hidden` group uses `<name>`'s floor, as it uses its plan.
ROUTE_PLAN_FLOORS = {
    "rover_3a_3poi_c3_implicit": 6.0,  # 6.08-6.24 of 10
    "rover_3a_3poi_c3_implicit_ord": 6.0,  # 6.08-6.24
    "rover_1a_3poi_c1_implicit": 6.0,  # 6.13-6.28 of 10
    "rover_1a_3poi_c1_implicit_ord": 6.0,  # 6.08-6.28
    "rover_6a_3poi_c3_implicit": 7.5,  # 7.79-7.98 of 10
    "rover_6a_3poi_c3_implicit_ord": 6.25,  # 6.36-6.52
    "rover_6a_6poi_c3": 4.25,  # 4.31-4.44 of 6, inner POIs need 100 coupled steps
    "rover_6a_6poi_c3_ord": 3.5,  # 3.59-3.66, inner POIs 1 and 2 never unlock
}


def test_every_rover_group_has_a_route_plan():
    missing = sorted(set(list_variants()) - set(ROUTE_PLANS))
    assert not missing, f"no ROUTE_PLANS entry for {missing}"


@pytest.mark.parametrize("variant", sorted(ROUTE_PLANS))
def test_every_route_plan_names_a_current_group_and_solves_it(variant):
    """A renamed or deleted group must take its plan along; every plan's team
    uses every POI up (or reaches its floor) and never abandons a POI."""
    assert variant in list_variants(), f"ROUTE_PLANS names {variant}, which has no conf/env file"
    env = load_rover_env(variant)
    keys = jax.vmap(jax.random.PRNGKey)(jnp.arange(2))
    traj, routes = planned_rollouts(env, keys, ROUTE_PLANS[variant])
    returns = np.asarray(traj.episode_return)
    floor = ROUTE_PLAN_FLOORS.get(variant.removesuffix("_hidden"))
    if floor is not None:
        assert np.all(returns >= floor)
    else:
        np.testing.assert_allclose(returns, float(env.pois.value.sum()), atol=1e-5)
        assert np.all(solve_steps(traj)[0] >= 0)
    assert_routed_rovers_stay_while_needed(env, traj, routes)


def test_plan_routes_fills_each_team_and_rejects_bad_plans():
    env = load_rover_env(ROUTED_VARIANT)
    teams = ROUTE_PLANS[ROUTED_VARIANT]
    _, state, _ = rollout_reset(env, jax.random.PRNGKey(0))
    routes = plan_routes(env, state, teams)
    counts = {tuple(route): n for route, n in zip(*np.unique(routes, axis=0, return_counts=True))}
    assert counts == {tuple(route): size for route, size in teams}
    with pytest.raises(ValueError, match="sum to 15"):
        plan_routes(env, state, [((0, 2), 3), ((0, 1), 4), ((3, 5), 4), ((3, 4), 4)])
    with pytest.raises(ValueError, match="fewer rovers"):
        plan_routes(env, state, [((0, 2), 4), ((0, 2), 4), ((3, 5), 4), ((3, 4), 4)])


def test_rollouts_are_deterministic_given_the_key():
    env = load_rover_env("rover_2poi_c2")
    run = jax.jit(lambda k: rollout(env, random_policy, k))
    a, b = run(jax.random.PRNGKey(7)), run(jax.random.PRNGKey(7))
    np.testing.assert_array_equal(np.asarray(a.positions), np.asarray(b.positions))


# ------------------------------------------------------------ visualization


def test_every_variant_writes_arena_and_trajectory_images(tmp_path):
    import matplotlib

    matplotlib.use("Agg")
    from environments.rover.visualize import plot_arena, plot_trajectory, save_figure

    for variant in list_variants():
        env = load_rover_env(variant)
        arena = save_figure(plot_arena(env, variant), tmp_path / f"{variant}.png")
        policy = make_scripted_coalition_policy(env)
        traj = jax.jit(lambda k: rollout(env, policy, k))(jax.random.PRNGKey(0))
        trajectory = save_figure(plot_trajectory(env, traj, variant), tmp_path / f"{variant}_t.png")
        for path in (arena, trajectory):
            assert path.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"


def test_figures_label_each_poi_order_only_in_an_ordered_layout():
    import matplotlib

    matplotlib.use("Agg")
    from environments.rover.visualize import plot_arena, plot_trajectory

    def labels(fig):
        texts = [t.get_text() for ax in fig.axes for t in ax.texts]
        ticks = [t.get_text() for ax in fig.axes for t in ax.get_yticklabels()]
        return texts + ticks

    cfg = load_env_config("rover_2poi_c2")
    plain = make_rover_env(cfg)
    assert not any("order" in s for s in labels(plot_arena(plain)))
    cfg["pois"]["entries"][0]["order"] = 1
    env = make_rover_env(cfg)
    assert "P0 · order 1" in labels(plot_arena(env))
    traj = rollout(env, make_scripted_coalition_policy(env), jax.random.PRNGKey(0))
    found = labels(plot_trajectory(env, traj))
    assert "P1 · c=2 · order 0" in found and "P0 (c=2, order 1)" in found
