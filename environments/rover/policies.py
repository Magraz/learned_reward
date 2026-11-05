"""Demonstration policies for the rover domain (visualization and tests only).

All follow the ``rollout`` signature ``policy_fn(params, key, obs, state)``.
The scripted policies read the privileged state (true POI positions), which a
learned policy must not:

* ``make_scripted_coalition_policy``: a generic team that fills POIs in index
  order and reassigns every rover whenever a POI is used up;
* ``make_route_policy``: each rover follows its own planned sequence of POIs
  (its route), passed as ``params``. ``plan_routes`` assigns rovers to the
  routes of a plan from the initial state, and ``planned_rollouts`` runs it.
  ``ROUTE_PLANS`` holds a hand-written plan for each rover env group.
"""

import jax
import jax.numpy as jnp
import numpy as np
from scipy.optimize import linear_sum_assignment

from environments.rover.base import wrap_angle
from environments.rover.pois import unlocked
from environments.rover.rollout import rollout, rollout_reset

# Fraction of a POI's radius a route rover parks at, measured from the centre.
# Inside the radius with a margin, and close enough to the edge that the next
# leg starts nearly at the boundary.
PARK_DEPTH = 0.9

# Team plans per rover env group (conf/env/<name>.yaml): ``(route, n_rovers)``
# pairs, a route being POI indices in visiting order. A rover waits at a locked
# POI (see ``pois.unlocked``) until it unlocks. Each plan was the best of the
# candidates tried on 2026-11-04 (whole-team sequences, splits and orders); it
# is not proven optimal. Steps to solve are ``scripted_steps`` medians.
_TWO_PAIRS = (((0,), 2), ((1,), 2))
_TWO_EIGHTS = (((0,), 8), ((1,), 8))
# Couplings sum to 32 = 2 x 16 rovers, so two waves: 8 rovers on each centre POI
# next to the spawn box, then 4 per corner, each corner team leaving from the
# nearer centre POI.
_C844844 = (
    ((0, 2), 4),  # (10, 12) -> (5, 17)
    ((0, 1), 4),  # (10, 12) -> (15, 17)
    ((3, 5), 4),  # (10, 8)  -> (5, 3)
    ((3, 4), 4),  # (10, 8)  -> (15, 3)
)
ROUTE_PLANS = {
    # One pair (or eight) per POI. Ordered, the second team parks at the locked
    # POI and starts the step after the first is used up.
    "rover_4a_2poi_c2": _TWO_PAIRS,
    "rover_4a_2poi_c2_ord": _TWO_PAIRS,
    "rover_4a_2poi_c2_implicit": _TWO_PAIRS,  # 30 and 60 coupled steps, in parallel
    "rover_16a_2poi_c8": _TWO_EIGHTS,
    "rover_16a_2poi_c8_ord": _TWO_EIGHTS,
    # One team (the coupling is more than half the rovers): only the order
    # 0 -> 1 -> 2 uses every POI up within the episode (see the group's yaml).
    "rover_2a_3poi_c2_implicit": (((0, 1, 2), 2),),
    "rover_4a_3poi_c3_implicit": (((0, 1, 2), 4),),
    # The coupling-4 corner needs every rover, so it goes first. Then three
    # rovers take the coupling-3 corner and one the coupling-1 corner, both next
    # to it; then two of the three take the coupling-2 corner, next to both.
    "rover_4a_4poi_c1234": (((3, 2, 1), 2), ((3, 2), 1), ((3, 0), 1)),
    # Strict order 0-1-2-3, one POI at a time. One rover works POI 0, then waits
    # at 3; a pair waits at 1, then joins the rover waiting at 2; those three
    # then join it at 3.
    "rover_4a_4poi_c1234_ord": (((0, 3), 1), ((1, 2, 3), 2), ((2, 3), 1)),
    # Two teams of 3, each working three neighbouring POIs of the ring.
    "rover_6a_6poi_c3": (((0, 1, 2), 3), ((3, 4, 5), 3)),
    # Strict order 0-...-5: the teams alternate, each driving to its next POI
    # (two along the ring) while the other team works the one in between.
    "rover_6a_6poi_c3_ord": (((0, 2, 4), 3), ((1, 3, 5), 3)),
    # Under `_ord` the left corners (order 2) wait for the right ones (order 1).
    "rover_16a_6poi_c844844": _C844844,
    "rover_16a_6poi_c844844_ord": _C844844,
    # Cannot collect the maximum 10: the POIs need 100, 150 and 60 coupled
    # steps, and two teams of 3 supply at most 200 in 100 steps. One team takes
    # POI 2 (0.1 per step), then POI 0; the other works POI 1 (0.02 per step)
    # for the whole episode.
    "rover_6a_3poi_c3_implicit": (((2, 0), 3), ((1,), 3)),
}
# Hiding changes only what the rovers sense, and these rovers read the true
# state, so each `_ord_hidden` group uses its `_ord` group's plan.
ROUTE_PLANS.update(
    {f"{name}_hidden": plan for name, plan in list(ROUTE_PLANS.items()) if name.endswith("_ord")}
)


def random_policy(params, key, obs, state):
    """Uniform random ``[turn, speed]`` actions."""
    return jax.random.uniform(key, (obs.shape[0], 2), minval=-1.0, maxval=1.0)


def _steer(env, pos, heading, waypoint):
    """Turn toward ``waypoint`` and drive at a speed that stops on it.

    Returns ``(turn, speed, dist)``: ``turn`` in [-1, 1] (0 once within 1e-3 of
    the waypoint); ``speed`` as a fraction of ``max_speed``, slowed while the
    rover still faces away after this step's turn; the distance to the waypoint.
    """
    delta = waypoint - pos
    dist = jnp.linalg.norm(delta, axis=-1)
    error = wrap_angle(jnp.arctan2(delta[:, 1], delta[:, 0]) - heading)
    turn = jnp.clip(error / env.max_turn, -1.0, 1.0)
    turn = jnp.where(dist < 1e-3, 0.0, turn)
    residual = error - turn * env.max_turn  # heading error after this step's turn
    speed = jnp.clip(dist / env.max_speed, 0.0, 1.0) * jnp.maximum(jnp.cos(residual), 0.0)
    return turn, speed, dist


def _unit(v):
    """Rows of ``v`` scaled to unit length; a zero row stays zero."""
    norm = jnp.linalg.norm(v, axis=-1, keepdims=True)
    return v / jnp.maximum(norm, 1e-9)


def _approach_point(centre, depth, pos):
    """The point ``depth`` from ``centre`` nearest ``pos``, or ``pos`` itself
    when the rover is already deeper than that."""
    offset = pos - centre
    dist = jnp.linalg.norm(offset, axis=-1, keepdims=True)
    return centre + jnp.minimum(dist, depth[:, None]) * _unit(offset)


def make_scripted_coalition_policy(env):
    """A scripted team that forms coalitions to meet each POI's coupling.

    Available POIs (budget left and unlocked, see ``pois.unlocked``) are
    filled in index order: rovers 0..c0-1 go to the first one, the next c1
    rovers to the second, and so on; rovers left over join the first available
    POI. When a POI is used up the assignment is recomputed, so a team too
    small to cover every POI at once covers them in sequence, and POIs with a
    consumption order are worked in that order.
    Each rover turns toward its target and drives at a speed that stops it on
    the target, slowing while it is still facing away.
    """

    def policy(params, key, obs, state):
        pois = state.pois
        pos, heading = state.rovers.position, state.rovers.heading
        available = (pois.remaining > 0) & unlocked(pois)
        need = jnp.where(available, pois.coupling, 0)
        cum = jnp.cumsum(need)  # (P,)
        rank = jnp.arange(pos.shape[0])
        first_block = jnp.argmax(rank[:, None] < cum[None, :], axis=1)  # first True
        target = jnp.where(rank < cum[-1], first_block, jnp.argmax(available))

        turn, speed, _ = _steer(env, pos, heading, pois.position[target])
        speed = jnp.where(available.any(), speed, 0.0)
        return jnp.stack([turn, 2.0 * speed - 1.0], axis=-1)

    return policy


def make_route_policy(env, park_depth=PARK_DEPTH):
    """Rovers that each follow a planned route of POIs.

    ``params`` is an ``(n_agents, K)`` int array: row r lists rover r's POIs in
    visiting order (pad a short route by repeating its last POI). A rover's
    current target is the first POI of its route with budget left, so the
    policy needs no memory: it advances when the target is used up.

    Toward its target a rover drives radially to the point ``park_depth *
    radius`` from the centre (the nearest such point, so it enters the radius
    as early as it can). Once inside it slides, still inside, to the point at
    that depth facing the route's next POI, and turns in place to face that
    POI, so the next leg starts at the edge and needs no turning. While its
    target still has budget it never makes a move that would leave the radius.
    With no POI left on its route it stops.
    """

    def policy(routes, key, obs, state):
        pois = state.pois
        pos, heading = state.rovers.position, state.rovers.heading
        rows = jnp.arange(pos.shape[0])
        left = pois.remaining[routes] > 0  # (A, K)
        k = jnp.argmax(left, axis=1)  # first route entry with budget left
        has_target = left.any(axis=1)
        later = left & (jnp.arange(routes.shape[1])[None, :] > k[:, None])
        has_next = later.any(axis=1)
        target = routes[rows, k]
        next_centre = pois.position[routes[rows, jnp.argmax(later, axis=1)]]

        centre, radius = pois.position[target], pois.radius[target]
        depth = park_depth * radius
        dist_centre = jnp.linalg.norm(pos - centre, axis=-1)
        inside = dist_centre <= radius
        park = centre + depth[:, None] * _unit(next_centre - centre)
        waypoint = jnp.where(
            (inside & has_next)[:, None], park, _approach_point(centre, depth, pos)
        )
        turn, speed, dist = _steer(env, pos, heading, waypoint)

        # Parked: face the next POI, so the next leg needs no turning.
        parked = dist < 1e-3
        face = next_centre - pos
        face_error = wrap_angle(jnp.arctan2(face[:, 1], face[:, 0]) - heading)
        turn = jnp.where(parked & has_next, jnp.clip(face_error / env.max_turn, -1.0, 1.0), turn)
        speed = jnp.where(parked, 0.0, speed)

        # Never leave the radius of a target that still needs this rover: if
        # the move (as env.move applies it) would end outside, only turn.
        new_heading = heading + turn * env.max_turn
        step = speed[:, None] * env.max_speed * jnp.stack(
            [jnp.cos(new_heading), jnp.sin(new_heading)], axis=-1
        )
        new_pos = jnp.clip(pos + step, 0.0, jnp.asarray(env.world_size))
        new_dist = jnp.linalg.norm(new_pos - centre, axis=-1)
        exits = inside & (new_dist > 0.999 * radius) & (new_dist > dist_centre)
        speed = jnp.where(exits, 0.0, speed)

        turn = jnp.where(has_target, turn, 0.0)
        speed = jnp.where(has_target, speed, 0.0)
        return jnp.stack([turn, 2.0 * speed - 1.0], axis=-1)

    return policy


def entry_steps(env, rovers, pois, park_depth=PARK_DEPTH):
    """``(n_agents, n_pois)`` steps each rover needs, alone, to get inside each
    POI's radius under ``make_route_policy``'s approach; ``inf`` if it cannot
    within ``env.max_steps``.

    Rovers do not interact, so a team's arrival time at a POI is the maximum of
    its members' entries.
    """
    n_agents, n_pois = rovers.position.shape[0], pois.position.shape[0]
    flat = jax.tree.map(lambda x: jnp.repeat(x, n_pois, axis=0), rovers)
    centre = jnp.tile(pois.position, (n_agents, 1))
    radius = jnp.tile(pois.radius, n_agents)
    depth = park_depth * radius

    def body(carry, t):
        rov, entered = carry
        turn, speed, _ = _steer(env, rov.position, rov.heading,
                                _approach_point(centre, depth, rov.position))
        rov, _ = env.move(rov, jnp.stack([turn, 2.0 * speed - 1.0], axis=-1))
        inside = jnp.linalg.norm(rov.position - centre, axis=-1) <= radius
        entered = jnp.where((entered == 0) & inside, t, entered)
        return (rov, entered), None

    entered0 = jnp.zeros(n_agents * n_pois, jnp.int32)
    (_, entered), _ = jax.lax.scan(body, (flat, entered0), jnp.arange(1, env.max_steps + 1))
    steps = jnp.where(entered > 0, entered, jnp.inf)
    return steps.reshape(n_agents, n_pois)


def plan_routes(env, state, teams, park_depth=PARK_DEPTH, entry=None):
    """Assign rovers to the routes of a team plan; returns ``(n_agents, K)``.

    ``teams``: ``(route, n_rovers)`` pairs (see ``ROUTE_PLANS``). The sizes must
    sum to ``n_agents``, and every POI must be visited by at least its coupling.

    The assignment minimizes the latest entry into a first POI (computed with
    ``entry_steps``, so it is exact for this controller): the first coupled
    step cannot come before every member of the team is inside. Among the
    assignments achieving that, it minimizes the summed distance from each
    rover to its parking point at its first POI, which sends rovers to the
    side of the POI facing their next leg. Runs on the host (scipy); pass a
    precomputed ``entry_steps`` array as ``entry`` to skip that call.
    """
    slots = [tuple(int(p) for p in route) for route, size in teams for _ in range(int(size))]
    if len(slots) != env.n_agents:
        raise ValueError(f"team sizes sum to {len(slots)}, env has {env.n_agents} rovers")
    coupling = np.asarray(state.pois.coupling)
    visits = np.zeros(len(coupling), int)
    for route in slots:
        visits[list(set(route))] += 1
    if np.any(visits < coupling):
        raise ValueError(
            f"POIs {np.flatnonzero(visits < coupling).tolist()} get fewer rovers "
            f"({visits[visits < coupling].tolist()}) than their coupling"
        )
    width = max(len(route) for route in slots)
    padded = np.asarray([route + route[-1:] * (width - len(route)) for route in slots])

    if entry is None:
        entry = entry_steps(env, state.rovers, state.pois, park_depth)
    cost = np.asarray(entry)[:, padded[:, 0]]  # (rovers, slots)
    position = np.asarray(state.pois.position)
    pos = np.asarray(state.rovers.position)
    first, second = position[padded[:, 0]], position[padded[:, min(1, width - 1)]]
    heading = second - first
    norm = np.linalg.norm(heading, axis=-1, keepdims=True)
    heading = np.where(norm > 0, heading / np.maximum(norm, 1e-9), 0.0)
    park = first + park_depth * np.asarray(state.pois.radius)[padded[:, 0], None] * heading
    slide = np.linalg.norm(pos[:, None, :] - park[None, :, :], axis=-1)

    def assign(limit):
        penalty = np.where(cost > limit, 1e6, 0.0) + slide
        rows, cols = linear_sum_assignment(penalty)
        return cols, bool(np.all(cost[rows, cols] <= limit))

    candidates = np.unique(cost[np.isfinite(cost)])
    if candidates.size == 0 or not assign(candidates[-1])[1]:
        raise ValueError("no assignment gets every rover into its first POI in time")
    lo, hi = 0, candidates.size - 1
    while lo < hi:  # smallest feasible latest-entry step
        mid = (lo + hi) // 2
        if assign(candidates[mid])[1]:
            hi = mid
        else:
            lo = mid + 1
    cols, _ = assign(candidates[lo])
    return padded[cols]


def planned_rollouts(env, keys, teams, park_depth=PARK_DEPTH):
    """For each key, plan routes from the initial state of ``rollout(env, _,
    key)`` and run ``make_route_policy`` on them.

    Returns ``(Trajectory, routes)`` with a leading ``len(keys)`` axis on both.
    The rollouts are one vmapped call; only the assignment loops on the host.
    """
    states = jax.jit(jax.vmap(lambda k: rollout_reset(env, k)[1]))(keys)
    entries = np.asarray(jax.jit(jax.vmap(
        lambda s: entry_steps(env, s.rovers, s.pois, park_depth)
    ))(states))
    states = jax.tree.map(np.asarray, states)
    routes = np.stack([
        plan_routes(env, jax.tree.map(lambda x: x[i], states), teams, park_depth, entries[i])
        for i in range(len(keys))
    ])
    policy = make_route_policy(env, park_depth)
    trajs = jax.jit(jax.vmap(lambda k, r: rollout(env, policy, k, r)))(keys, jnp.asarray(routes))
    return trajs, routes
