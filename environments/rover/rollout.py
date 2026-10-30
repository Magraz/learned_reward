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

import jax
import jax.numpy as jnp


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


def rollout_view_policy(env, policy, key):
    """Record one episode using a runner's stateful deterministic view policy.

    The host loop lets recurrent policies carry hidden state between decisions.
    Reset that state for every episode and stop at termination or truncation.
    Unlike ``rollout``, this helper must not be wrapped in ``jax.jit``.
    """
    policy.reset()
    obs, state = jax.jit(env.reset)(key)
    step = jax.jit(env.step)
    positions = [state.rovers.position]
    headings = [state.rovers.heading]
    remaining = [state.pois.remaining]
    actions, rewards, dones = [], [], []
    for _ in range(env.max_steps):
        action = policy(obs)
        obs, state, _, terminated, truncated, info = step(state, action)
        done = terminated | truncated
        positions.append(state.rovers.position)
        headings.append(state.rovers.heading)
        remaining.append(state.pois.remaining)
        actions.append(action)
        rewards.append(info["task_reward"])
        dones.append(done)
        if bool(done):
            break
    rewards = jnp.stack(rewards)
    return Trajectory(
        positions=jnp.stack(positions),
        headings=jnp.stack(headings),
        poi_remaining=jnp.stack(remaining),
        actions=jnp.stack(actions),
        rewards=rewards,
        done=jnp.stack(dones),
        episode_return=rewards.sum(),
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
