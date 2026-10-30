"""Count the steps the planned scripted team needs to solve a rover variant.

Runs ``policies.planned_rollouts`` (``make_route_policy`` on the routes
``plan_routes`` assigns from ``ROUTE_PLANS[variant]``) over many reset seeds
and prints how often every POI is used up, and in how many steps::

    uv run python -m environments.rover.scripted_steps \\
        [--variant rover_6poi_c8-4-4-8-4-4] [--n-seeds 1000] [--figures 2]

Seed k is ``jax.random.PRNGKey(k)``, the same episode ``visualize`` draws for
``--seeds k``. ``--figures N`` saves the first N episodes as
``<out-dir>/trajectories/<variant>_routed_seed<k>.png``.
"""

import argparse
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from environments.rover.config import REPO_ROOT, load_rover_env
from environments.rover.policies import ROUTE_PLANS, planned_rollouts


def solve_steps(traj):
    """Per episode: the step at which every POI was used up, or -1 if the
    episode was truncated first. Also returns ``(n, n_pois)`` used-up steps
    (-1 = never)."""
    remaining = np.asarray(traj.poi_remaining)  # (n, T+1, P), index 0 = reset
    used = remaining <= 0
    used_at = np.where(used.any(axis=1), used.argmax(axis=1), -1)
    solved = used[:, -1].all(axis=1)
    steps = np.where(solved, used_at.max(axis=1), -1)
    return steps, used_at


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--variant", default="rover_6poi_c8-4-4-8-4-4", choices=sorted(ROUTE_PLANS))
    parser.add_argument("--n-seeds", type=int, default=1000)
    parser.add_argument("--first-seed", type=int, default=0)
    parser.add_argument("--figures", type=int, default=0,
                        help="save trajectory figures of the first N seeds")
    parser.add_argument("--out-dir", type=Path, default=REPO_ROOT / "plotting" / "rover")
    args = parser.parse_args(argv)

    env = load_rover_env(args.variant)
    teams = ROUTE_PLANS[args.variant]
    seeds = np.arange(args.first_seed, args.first_seed + args.n_seeds)
    keys = jax.vmap(jax.random.PRNGKey)(jnp.asarray(seeds))
    traj, routes = planned_rollouts(env, keys, teams)
    steps, used_at = solve_steps(traj)
    returns = np.asarray(traj.episode_return)
    solved = steps >= 0

    print(f"{args.variant}: {env.n_agents} rovers, {env.n_pois} POIs, max_steps {env.max_steps}")
    print("plan: " + ", ".join(f"{size} x {' -> '.join(map(str, route))}" for route, size in teams))
    print(f"seeds {seeds[0]}..{seeds[-1]}: solved {solved.sum()} / {len(seeds)}, "
          f"return mean {returns.mean():.4f} (max {float(env.pois.value.sum()):.1f})")
    if solved.any():
        s = steps[solved]
        print(f"steps to solve: min {s.min()}, median {np.median(s):.0f}, "
              f"mean {s.mean():.2f}, max {s.max()}")
        values, counts = np.unique(s, return_counts=True)
        print("  distribution: " + ", ".join(f"{v}: {c}" for v, c in zip(values, counts)))
        print("  POI used up at step (median): " + ", ".join(
            f"POI {p} {np.median(used_at[solved, p]):.0f}" for p in range(env.n_pois)))

    if args.figures:
        import matplotlib

        matplotlib.use("Agg")
        from environments.rover.visualize import plot_trajectory, save_figure

        for i, seed in enumerate(seeds[: args.figures]):
            one = jax.tree.map(lambda x: x[i], traj)
            title = f"{args.variant} · routed policy · seed {seed}"
            out = args.out_dir / "trajectories" / f"{args.variant}_routed_seed{seed}.png"
            print(save_figure(plot_trajectory(env, one, title), out))


if __name__ == "__main__":
    main()
