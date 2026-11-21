"""Critic value maps for the rover domain: what a trained critic predicts over the arena.

A critic's value V(s) is a function of the whole state, not of a position:
every rover's position and heading, the POI budgets, and a recurrent critic's
memory of the episode (its hidden state). A 2-D map is therefore a choice of
what to hold fixed. Two are provided:

* **Slices** (`CriticProbe.team_slice`): take one real state of an episode and
  the critic's hidden state carried into it, translate the whole team as a
  block so its centroid sits on each cell of a grid (positions clipped to the
  arena, as ``BaseRoverEnv.move`` clips them), and evaluate the critic there.
  A critic that reads the heading-relative sensor sectors is averaged over
  ``n_headings`` headings, the team turning as a block (every heading gets the
  same offset); a critic that reads the env's global state never sees the
  heading and is evaluated once. Away from the states the critic trained on
  this is extrapolation, and a recurrent critic's history still says the team
  was elsewhere: the slice teleports the team.
* **Visitation** (`visitation_maps`): the critic's value at the states that
  episodes actually visit, with its true history, averaged per cell by the
  team centroid; next to it the realized discounted return-to-go, averaged the
  same way, and the visits per cell.

Values are in reward units: a runner's ``_view_critic`` undoes any value
normalization. A per-rover value head (difference rewards) is averaged over
the rovers (`team_value`).

Algorithm-agnostic: everything takes ``value_fn(states, h, mask) -> (values
(B, n_outputs), h)`` over a batch of ``B`` env states, as ``rollout`` takes a
``policy_fn``; ``MAPPO_JAX_Runner._view_critic`` builds it for both stacks.
``view=true`` draws both maps (``visualize.plot_value_slices`` and
``visualize.plot_value_visitation``).
"""

from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from environments.rover.base import wrap_angle
from environments.rover.state import RoverEnvState, RoverState

N_HEADINGS = 16  # headings averaged per cell for a critic that reads the sectors
SLICE_CELLS = 80  # slice grid cells on the arena's longer side
VISITATION_CELLS = 40  # visitation grid cells on the arena's longer side
MIN_VISITS = 3  # a visitation cell with fewer visited states is left blank


def team_value(values):
    """One value per state from a critic head ``(..., n_outputs)``: the team
    value, or the mean over the rovers of a per-rover head. With coupling equal
    to the team size every rover's difference reward is the team reward, so
    that mean is the team value there too."""
    return values.mean(axis=-1)


def grid_axes(env, n_cells):
    """Cell centres ``(xs (nx,), ys (ny,))`` of a grid over the arena with
    ``n_cells`` cells on its longer side and near-square cells."""
    width, height = env.world_size
    cell = max(width, height) / n_cells
    nx, ny = max(1, round(width / cell)), max(1, round(height / cell))
    return (np.arange(nx) + 0.5) * width / nx, (np.arange(ny) + 0.5) * height / ny


def state_at(traj, t):
    """The env state at step ``t`` of a ``rollout_view_policy`` trajectory."""
    return jax.tree.map(lambda x: x[t], traj.states)


def discounted_return_to_go(rewards, gamma):
    """``G (T+1,)``: ``G[t] = sum_k gamma^k rewards[t + k]``, the discounted
    team return from state ``t`` (``rewards[t]`` is paid on the step out of
    state ``t``), and ``G[T] = 0``.

    No bootstrap at the end of the episode: ``rmappo_jax`` treats time limits
    as terminal, so this is exactly its critic's target. ``mappo_jax``
    bootstraps at a time limit, so near the end its critic should exceed it.
    """
    rewards = np.asarray(rewards, np.float64)
    returns = np.zeros(rewards.shape[0] + 1)
    for t in range(rewards.shape[0] - 1, -1, -1):
        returns[t] = rewards[t] + gamma * returns[t + 1]
    return returns


def phase_steps(traj, max_panels=6, min_panels=3):
    """``[(step, label)]``: the states of an episode to slice, in step order.

    Step 0, then the first step each POI reads used up (several POIs used up
    on one step share it). Only states the critic is queried at in training
    count, i.e. steps ``0 .. T-1``: the state after the last step is never
    evaluated (the env restarts), so a POI used up on the final step adds no
    panel. More than ``max_panels``: the depletions are subsampled evenly.
    Fewer than ``min_panels``: evenly spaced steps are added, each the
    candidate farthest from the steps already chosen.
    """
    remaining = np.asarray(traj.poi_remaining)
    last = int(np.asarray(traj.rewards).shape[0]) - 1
    used_at = {}
    for poi in range(remaining.shape[1]):
        zero = np.flatnonzero(remaining[:, poi] <= 0)
        if zero.size and 0 < zero[0] <= last:
            used_at.setdefault(int(zero[0]), []).append(poi)
    events = sorted(used_at)
    if len(events) > max_panels - 1:
        keep = np.unique(np.round(np.linspace(0, len(events) - 1, max_panels - 1)))
        events = [events[int(i)] for i in keep]
    labels = {0: "start"}
    for t in events:
        labels[t] = " + ".join(f"P{p}" for p in used_at[t]) + " used up"
    steps = [0, *events]
    candidates = np.unique(np.round(np.linspace(0, last, 2 * min_panels + 1)).astype(int))
    while len(steps) < min(min_panels, last + 1):
        distance = np.abs(candidates[:, None] - np.asarray(steps)[None]).min(axis=1)
        if distance.max() == 0:
            break
        new = int(candidates[np.argmax(distance)])
        steps.append(new)
        labels[new] = ""
    return [(t, labels[t]) for t in sorted(steps)]


class CriticProbe:
    """Jitted evaluations of one trained critic over rover env states.

    Args:
        env: the rover env the critic trained on.
        value_fn: ``value_fn(states, h, mask (B,)) -> (values (B, n_outputs),
            h)`` over a batch of ``B`` env states, values in reward units
            (``MAPPO_JAX_Runner._view_critic``).
        initial_hidden: ``initial_hidden(B)``, the hidden state an episode
            starts from (zeros, or a scalar placeholder for a feedforward
            critic).
        n_headings: headings averaged per cell; default 1 when the critic reads
            the env's global state (no heading in it), else ``N_HEADINGS``.
        chunk_size: grid cells evaluated at once by a slice, to bound memory.
    """

    def __init__(self, env, value_fn, initial_hidden, n_headings=None, chunk_size=1024):
        self.env = env
        if n_headings is None:
            n_headings = 1 if hasattr(env, "global_state") else N_HEADINGS
        self.n_headings = int(n_headings)
        self.n_steps = env.max_steps + 1
        h0 = initial_hidden(1)
        # A recurrent critic carries a hidden state; a feedforward one passes a
        # scalar placeholder.
        self.recurrent = bool(jnp.ndim(h0) > 0)
        size = jnp.asarray(env.world_size, jnp.float32)
        offsets = 2 * jnp.pi * jnp.arange(self.n_headings) / self.n_headings
        ones = jnp.ones((self.n_headings,))

        @jax.jit
        def along(states):
            def body(h, state):
                batched = jax.tree.map(lambda x: x[None], state)
                values, h_next = value_fn(batched, h, jnp.ones((1,)))
                return h_next, (team_value(values)[0], h)

            _, (values, hidden) = jax.lax.scan(body, h0, states)
            return values, hidden

        @jax.jit
        def values_at(state, h, points):
            rovers = state.rovers
            shift = -rovers.position.mean(axis=0)
            h = h if h.ndim == 0 else jnp.broadcast_to(h, (self.n_headings, *h.shape[1:]))

            def at_point(point):
                position = jnp.clip(rovers.position + shift + point, 0.0, size)
                heading = wrap_angle(rovers.heading[None] + offsets[:, None])
                states = RoverEnvState(
                    rovers=RoverState(
                        position=jnp.broadcast_to(
                            position, (self.n_headings, *position.shape)
                        ),
                        heading=heading,
                    ),
                    pois=jax.tree.map(
                        lambda x: jnp.broadcast_to(x, (self.n_headings, *x.shape)),
                        state.pois,
                    ),
                    t=jnp.broadcast_to(state.t, (self.n_headings,)),
                )
                values, _ = value_fn(states, h, ones)
                return team_value(values).mean()

            return jax.lax.map(at_point, points, batch_size=chunk_size)

        self._along = along
        self._values_at = values_at

    def along(self, traj):
        """``(values (T+1,), hidden)`` along a ``rollout_view_policy``
        trajectory: the critic run over its states in order with mask 1, from
        the episode's initial hidden state, as in training. ``hidden[t]`` is
        the hidden state carried into step ``t`` (what a slice of that step
        starts from). Padded to ``max_steps + 1`` states, so every episode
        reuses one compilation."""
        if traj.states is None:
            raise ValueError("the trajectory has no recorded states (use rollout_view_policy)")
        n = int(np.asarray(traj.poi_remaining).shape[0])
        padded = jax.tree.map(
            lambda x: jnp.concatenate([x, jnp.repeat(x[-1:], self.n_steps - n, axis=0)]),
            traj.states,
        )
        values, hidden = self._along(padded)
        return np.asarray(values[:n]), jax.tree.map(lambda x: x[:n], hidden)

    def values_at(self, state, h, points):
        """``(N,)`` values with the team translated so its centroid sits on
        each of ``points (N, 2)``, averaged over the probe's headings."""
        return np.asarray(self._values_at(state, h, jnp.asarray(points, jnp.float32)))

    def team_slice(self, state, h, xs, ys):
        """``(ny, nx)`` values over the grid with cell centres ``xs`` and ``ys``
        (row 0 = smallest y, for ``imshow(origin="lower")``)."""
        grid_x, grid_y = np.meshgrid(xs, ys)
        points = np.stack([grid_x.ravel(), grid_y.ravel()], axis=1)
        return self.values_at(state, h, points).reshape(len(ys), len(xs))

    def visited(self, traj, gamma):
        """``(centroids (T, 2), values (T,), returns (T,))`` at every state of
        an episode the critic is queried at in training (steps ``0 .. T-1``):
        the team centroid, the critic's value with its true history, and the
        realized discounted return-to-go."""
        values, _ = self.along(traj)
        returns = discounted_return_to_go(traj.rewards, gamma)
        n = int(np.asarray(traj.rewards).shape[0])
        centroids = np.asarray(traj.positions)[:n].mean(axis=1)
        return centroids, values[:n], returns[:n]


class VisitationMaps(NamedTuple):
    xs: np.ndarray  # (nx,) cell centres
    ys: np.ndarray  # (ny,)
    mean_value: np.ndarray  # (ny, nx) mean critic value, NaN under min_visits
    mean_return: np.ndarray  # (ny, nx) mean realized discounted return-to-go
    visits: np.ndarray  # (ny, nx) visited states per cell
    min_visits: int


def visitation_maps(env, centroids, values, returns, n_cells=VISITATION_CELLS,
                    min_visits=MIN_VISITS):
    """Bin visited states by team centroid ``(N, 2)``: the mean critic value
    and mean realized return per cell (NaN in a cell with fewer than
    ``min_visits`` states) and the visits per cell."""
    xs, ys = grid_axes(env, n_cells)
    width, height = env.world_size
    edges = [np.linspace(0, height, len(ys) + 1), np.linspace(0, width, len(xs) + 1)]
    centroids = np.asarray(centroids, np.float64)

    def binned(weights=None):
        counts, _, _ = np.histogram2d(
            centroids[:, 1], centroids[:, 0], bins=edges, weights=weights
        )
        return counts

    visits = binned()
    enough = visits >= min_visits
    with np.errstate(invalid="ignore", divide="ignore"):
        mean_value = np.where(enough, binned(np.asarray(values)) / visits, np.nan)
        mean_return = np.where(enough, binned(np.asarray(returns)) / visits, np.nan)
    return VisitationMaps(xs, ys, mean_value, mean_return, visits, int(min_visits))
