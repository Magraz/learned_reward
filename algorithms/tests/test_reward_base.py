"""`reward_base`: the evaluation return in the unmodified env (POI order removed).

Covers `environments.rover.config.base_env_config`, the stats tracker's
`reward_base` series, and both stacks' train loops writing it with the length
of `reward`.
"""

import copy
import pickle
import warnings

import jax
import numpy as np
import pytest
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra

import train
from algorithms.algorithms import _dispatch
from algorithms.mappo_jax.stats_tracker import TrainingStatsTracker
from environments.rover.config import (
    CONF_DIR,
    base_env_config,
    load_env_config,
    make_rover_env,
)

CPU = jax.devices("cpu")[0]


# ------------------------------------------------------------ base_env_config


@pytest.mark.parametrize("group", ["rover_4a_2poi_c2", "rover_3a_3poi_c3_implicit"])
def test_a_layout_with_one_order_has_no_base_config(group):
    assert base_env_config(load_env_config(group)) is None


@pytest.mark.parametrize(
    "group, base",
    [
        ("rover_4a_2poi_c2_ord", "rover_4a_2poi_c2"),
        ("rover_4a_2poi_c2_ord_hidden", "rover_4a_2poi_c2"),
        ("rover_4a_4poi_c1234_hidden", "rover_4a_4poi_c1234"),
    ],
)
def test_the_base_config_builds_the_unordered_group(group, base):
    with jax.default_device(CPU):
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            env = make_rover_env(base_env_config(load_env_config(group)))
        expected = make_rover_env(load_env_config(base))
        key = jax.random.PRNGKey(3)
        obs, state = env.reset(key)
        expected_obs, expected_state = expected.reset(key)
    np.testing.assert_array_equal(obs, expected_obs)
    # Every field of the reset state (POI order and flags included) matches.
    jax.tree_util.tree_map_with_path(
        lambda path, a, b: np.testing.assert_array_equal(
            a, b, err_msg=jax.tree_util.keystr(path)
        ),
        state,
        expected_state,
    )


def test_the_base_config_keeps_the_runs_own_sensing():
    # rover_16a_6poi_c844844_ord senses globally, unlike rover_16a_6poi_c844844
    # (4.0): the base env removes the order only.
    cfg = load_env_config("rover_16a_6poi_c844844_ord")
    base = base_env_config(cfg)
    assert base["observation"] == cfg["observation"]
    assert base["observation"]["sensor_radius"] is None
    assert all("order" not in entry for entry in base["pois"]["entries"])


def test_the_base_config_leaves_its_input_unchanged():
    cfg = load_env_config("rover_4a_4poi_c1234_ord_hidden")
    before = copy.deepcopy(cfg)
    base_env_config(cfg)
    assert cfg == before


# ------------------------------------------------------------- stats tracker


def test_a_resumed_run_without_reward_base_is_padded_to_full_length():
    old = {
        "total_steps": [10, 20],
        "reward": [1.0, 2.0],
        "episodes": [1, 2],
        "training_time": [0.1, 0.2],
        "collection_time": [0.0, 0.0],
        "update_time": [0.0, 0.0],
        "eval_time": [0.0, 0.0],
    }
    tracker = TrainingStatsTracker()
    tracker.load_from_dict(old)
    tracker.initialize_for_train(checkpoint=True)
    for i in range(2):
        tracker.record_iteration(
            steps_completed=30 + 10 * i,
            episodes_completed=3 + i,
            reward=3.0,
            reward_base=4.0,
            collection_time=0.0,
            update_time=0.0,
            eval_time=0.0,
        )
    stats = tracker.to_dict()
    assert len(stats["reward_base"]) == len(stats["reward"]) == 4
    assert np.isnan(stats["reward_base"][:2]).all()
    assert stats["reward_base"][2:] == [4.0, 4.0]


# ----------------------------------------------------------------- end to end

_TINY = [
    "trial_id=smoke",
    "env.n_envs=2",
    "env.max_steps=10",
    "params.n_steps=10",
    "params.n_total_steps=60",  # 3 updates of 10 steps x 2 envs
    "params.n_eval_episodes=4",
    "params.n_minibatches=1",
]
_STACKS = {
    # mappo_jax's model group (conf/model/mlp.yaml) is missing, so it runs on
    # the algorithm group's own model_params.
    "mappo_jax": ["algorithm=mappo_jax", "~model"],
    "rmappo_jax": [
        "algorithm=rmappo_jax", "model=mappo_mlp", "params.eval_interval=1",
    ],
}


def _train(stack, group, tmp_path):
    if GlobalHydra.instance().is_initialized():
        GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir=str(CONF_DIR), version_base=None):
        cfg = compose(
            config_name="config", overrides=[f"env={group}", *_STACKS[stack], *_TINY]
        )
    args = train._build_dispatch_args(cfg, {"env": group, "model": stack})
    args["results_dir"] = tmp_path
    with jax.default_device(CPU):
        _dispatch(**args)
    with open(tmp_path / "smoke" / "logs" / "training_stats_finished.pkl", "rb") as f:
        return pickle.load(f)


@pytest.mark.parametrize("stack", sorted(_STACKS))
def test_an_unordered_run_logs_reward_base_as_a_copy_of_reward(stack, tmp_path):
    stats = _train(stack, "rover_4a_2poi_c2", tmp_path)
    assert len(stats["total_steps"]) == 3
    assert stats["reward_base"] == stats["reward"]


@pytest.mark.parametrize("stack", sorted(_STACKS))
def test_an_ordered_run_logs_reward_base_of_full_length(stack, tmp_path):
    stats = _train(stack, "rover_4a_2poi_c2_ord", tmp_path)
    reward, reward_base = np.asarray(stats["reward"]), np.asarray(stats["reward_base"])
    assert len(reward_base) == len(reward) == len(stats["total_steps"]) == 3
    assert np.isfinite(reward_base).all()
    # Without hiding, the observation is the same in both envs, so the rovers
    # drive the same paths, and the unordered env pays on a superset of steps.
    assert (reward_base >= reward - 1e-5).all()
