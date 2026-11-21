"""Algorithm-agnostic episode rollout for the rover domain.

``rollout`` runs one episode as a ``lax.scan`` of ``max_steps`` steps. After
the episode ends the state is frozen and rewards are zero, so the scan length
is static while episodes stay correct. It is pure, so the same call serves:

* visualization / debugging (one episode, inspect the ``Trajectory``);
* RL-style evaluation, vmapped over reset keys;
* evolutionary fitness, vmapped over policy parameters::

    fitness = jax.jit(jax.vmap(
        lambda p, k: rollout(env, policy_fn, k, p).episode_return
    ))(population_params, keys)

``policy_fn(params, key, obs, state) -> (n_agents, 2)`` actions in [-1, 1].
It receives the full state so scripted (privileged) policies can be expressed;
a learned policy should read only ``obs``.
"""

import dataclasses
import weakref

import jax
import jax.numpy as jnp
import numpy as np


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class Trajectory:
    positions: jax.Array  # (T+1, A, 2), index 0 = reset
    headings: jax.Array  # (T+1, A)
    poi_remaining: jax.Array  # (T+1, P)
    actions: jax.Array  # (T, A, 2)
    rewards: jax.Array  # (T,) team reward (info["task_reward"], not step's per-rover reward)
    done: jax.Array  # (T,) bool, episode over after step t (stays True)
    episode_return: jax.Array  # () sum of the team rewards
    # (T, P) bool: the POI mask the policy's observation was built under at
    # each step (the rmappo_jax POI manager's), or None for a policy without
    # one. Only `rollout_view_policy` records it.
    poi_masks: jax.Array | None = None
    # The env state of every step: a `RoverEnvState` whose leaves have a
    # leading T+1 axis (index 0 = reset), from which any learner's input can be
    # rebuilt exactly (the critic value maps, `value_map.py`). Only
    # `rollout_view_policy` records it; None from `rollout`.
    states: object | None = None


# The jitted `reset` and `step` of each env `rollout_view_policy` has run, so
# repeated episodes reuse one trace (a fresh `jax.jit(env.step)` re-traces).
_JITTED = weakref.WeakKeyDictionary()


def _jitted(env):
    if env not in _JITTED:
        _JITTED[env] = (jax.jit(env.reset), jax.jit(env.step))
    return _JITTED[env]


def _stack(arrays):
    """Stack per-step arrays on the host, then move them in one transfer
    (``jnp.stack`` of a hundred small device arrays costs a dispatch each)."""
    return jnp.asarray(np.stack([np.asarray(a) for a in arrays]))


def rollout_view_policy(env, policy, key):
    """Record one episode using a runner's stateful view policy.

    The host loop lets recurrent policies carry hidden state between decisions.
    Reset that state for every episode and stop at termination or truncation.
    Unlike ``rollout``, this helper must not be wrapped in ``jax.jit``.

    The policy is called as ``policy(obs, state=state)``; a policy that ignores
    ``obs`` and rebuilds it from the state (the POI manager's) needs the state.
    A policy with a ``last_poi_mask`` attribute that is not None after the call
    has its masks recorded in ``Trajectory.poi_masks``. Every env state is
    recorded in ``Trajectory.states``; positions, headings and budgets are
    read from them.
    """
    policy.reset()
    reset, step = _jitted(env)
    obs, state = reset(key)
    states = [state]
    actions, rewards, dones, poi_masks = [], [], [], []
    for _ in range(env.max_steps):
        action = policy(obs, state=state)
        poi_mask = getattr(policy, "last_poi_mask", None)
        if poi_mask is not None:
            poi_masks.append(poi_mask)
        obs, state, _, terminated, truncated, info = step(state, action)
        done = terminated | truncated
        states.append(state)
        actions.append(action)
        rewards.append(info["task_reward"])
        dones.append(done)
        if bool(done):
            break
    states = jax.tree.map(lambda *leaves: _stack(leaves), *states)
    rewards = _stack(rewards)
    return Trajectory(
        positions=states.rovers.position,
        headings=states.rovers.heading,
        poi_remaining=states.pois.remaining,
        actions=_stack(actions),
        rewards=rewards,
        done=_stack(dones),
        episode_return=rewards.sum(),
        poi_masks=_stack(poi_masks) if poi_masks else None,
        states=states,
    )


def rollout_reset(env, key):
    """``(obs, state, policy_key)`` that ``rollout(env, _, key)`` starts from.

    Lets a planner read the initial state and pass its plan as ``params``
    (see ``policies.planned_rollouts``).
    """
    k_reset, k_policy = jax.random.split(key)
    obs, state = env.reset(k_reset)
    return obs, state, k_policy


def rollout(env, policy_fn, key, params=None, n_steps=None):
    n_steps = env.max_steps if n_steps is None else int(n_steps)
    obs, state, k_policy = rollout_reset(env, key)

    def body(carry, k):
        obs, state, done = carry
        actions = policy_fn(params, k, obs, state)
        n_obs, n_state, _, terminated, truncated, info = env.step(state, actions)
        n_state = jax.tree.map(lambda new, old: jnp.where(done, old, new), n_state, state)
        n_obs = jnp.where(done, obs, n_obs)
        reward = jnp.where(done, 0.0, info["task_reward"])
        n_done = done | terminated | truncated
        record = (
            n_state.rovers.position,
            n_state.rovers.heading,
            n_state.pois.remaining,
            actions,
            reward,
            n_done,
        )
        return (n_obs, n_state, n_done), record

    keys = jax.random.split(k_policy, n_steps)
    _, (pos, heading, remaining, actions, rewards, done) = jax.lax.scan(
        body, (obs, state, jnp.zeros((), bool)), keys
    )
    return Trajectory(
        positions=jnp.concatenate([state.rovers.position[None], pos]),
        headings=jnp.concatenate([state.rovers.heading[None], heading]),
        poi_remaining=jnp.concatenate([state.pois.remaining[None], remaining]),
        actions=actions,
        rewards=rewards,
        done=done,
        episode_return=rewards.sum(),
    )
