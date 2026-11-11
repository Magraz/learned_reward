"""Tile the trained-policy trajectory figures in the order of `multi_batches`.

`train.py ... view=true` (or `scripts/view_results.sh`) writes, for each run,
`<base_path>/<batch>/<experiment>/<trial>/logs/rover/trajectory_seed<k>.png`:
three deterministic episodes, at seeds = the trial's seed + 1, 2, 3.

This script reads `plotting/config.yaml` (`base_path`, `multi_batches`,
`experiments`, `trials`, `max_rew_possible`, `custom_labels`) and, for every
experiment and trial, writes one figure per episode: one panel per batch of
`multi_batches`, in that order, filled row by row with ceil(sqrt(n)) columns,
the layout of the notebook's multi-batch reward grid. The k-th episode of a
trial has the same seed in every run, since a trial's seed comes from the
shared seed list.

    uv run python plotting/plot_learned_trajectories.py
    uv run python plotting/plot_learned_trajectories.py --experiments rmappo --trials 0 --episodes 1

Output: `<out-dir>/<experiment>_trial<t>_episode<k>.png` (default out-dir
`plotting/rover/learned_trajectories`). A run without the figure gets an empty
panel; a figure older than its run's saved model is flagged, because it shows
an earlier policy (re-run `view=true` on that run).
"""

import argparse
import math
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import yaml

PLOTTING_DIR = Path(__file__).resolve().parent
TRAJECTORY_PATTERN = re.compile(r"trajectory_seed(\d+)\.png$")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--config", type=Path, default=PLOTTING_DIR / "config.yaml")
    parser.add_argument(
        "--experiments", nargs="+", help="default: config experiments"
    )
    parser.add_argument(
        "--trials",
        nargs="+",
        help="default: the config trials that have at least one figure on disk",
    )
    parser.add_argument(
        "--episodes",
        nargs="+",
        type=int,
        default=[1, 2, 3],
        help="1-based episode index within a run, in seed order (default: 1 2 3)",
    )
    parser.add_argument(
        "--columns", type=int, help="default: ceil(sqrt(number of batches))"
    )
    parser.add_argument(
        "--out-dir", type=Path, default=PLOTTING_DIR / "rover" / "learned_trajectories"
    )
    parser.add_argument(
        "--dpi", type=int, default=220, help="220 keeps each panel near its native width"
    )
    return parser.parse_args()


def run_dir(base_path, batch, experiment, trial):
    return Path(base_path) / batch / experiment / str(trial)


def trajectory_figures(run):
    """The run's trajectory PNGs as [(seed, path)], in seed order."""
    figures = []
    for path in (run / "logs" / "rover").glob("trajectory_seed*.png"):
        match = TRAJECTORY_PATTERN.search(path.name)
        if match:
            figures.append((int(match.group(1)), path))
    return sorted(figures)


def saved_model(run):
    """The model `view` loads (finished, else checkpoint), or None."""
    for name in ("models_finished.msgpack", "models_checkpoint.msgpack"):
        path = run / "models" / name
        if path.is_file():
            return path
    return None


def grid_shape(n_panels, n_cols=None):
    """(n_rows, n_cols); ceil(sqrt(n)) columns, as the notebook's make_multi_batch_grid."""
    n_cols = n_cols or math.ceil(math.sqrt(n_panels))
    return math.ceil(n_panels / n_cols), n_cols


def plot_episode_grid(batches, panels, title, max_rewards, n_cols, out_path, dpi):
    """One figure: panels[i] = (seed, path, stale) or None for batches[i]."""
    images = [plt.imread(p[1]) if p else None for p in panels]
    height, width = next(img.shape[:2] for img in images if img is not None)
    n_rows, n_cols = grid_shape(len(batches), n_cols)
    panel_width = 8.0
    panel_height = panel_width * height / width + 0.5  # room for the panel title
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(panel_width * n_cols, panel_height * n_rows + 0.6),
        squeeze=False,
        layout="constrained",
    )
    for ax in axes.flat:
        ax.axis("off")
    for ax, batch, panel, image in zip(axes.flat, batches, panels, images):
        max_reward = max_rewards.get(batch)
        lines = [batch]
        if max_reward is not None:
            lines.append(f"best scripted return {max_reward:g}")
        if panel and panel[2]:
            lines.append("⚠ figure older than the saved model")
        ax.set_title("\n".join(lines), fontsize=12, loc="left")
        if image is None:
            ax.text(0.5, 0.5, "no figure", ha="center", va="center", fontsize=14,
                    color="0.5", transform=ax.transAxes)
        else:
            ax.imshow(image)
    fig.suptitle(title, fontsize=16, x=0.0, ha="left")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)


def main():
    args = parse_args()
    with args.config.open() as f:
        config = yaml.safe_load(f)
    base_path = config["base_path"]
    batches = config.get("multi_batches") or []
    if not batches:
        raise SystemExit(f"no multi_batches in {args.config}")
    experiments = args.experiments or config["experiments"]
    trials = args.trials or [str(t) for t in config["trials"]]
    max_rewards = config.get("max_rew_possible") or {}
    labels = config.get("custom_labels") or {}

    written, missing, stale = [], [], []
    for experiment in experiments:
        for trial in trials:
            runs = [run_dir(base_path, b, experiment, trial) for b in batches]
            figures = [trajectory_figures(run) for run in runs]
            if not any(figures):
                if args.trials:
                    print(f"skipped {experiment} trial {trial}: no figures")
                continue
            for episode in args.episodes:
                panels = []
                for batch, run, run_figures in zip(batches, runs, figures):
                    if episode > len(run_figures):
                        panels.append(None)
                        missing.append(f"{batch}/{experiment}/{trial} episode {episode}")
                        continue
                    seed, path = run_figures[episode - 1]
                    model = saved_model(run)
                    is_stale = model is not None and path.stat().st_mtime < model.stat().st_mtime
                    if is_stale:
                        stale.append(str(path.relative_to(base_path)))
                    panels.append((seed, path, is_stale))
                seeds = sorted({p[0] for p in panels if p})
                seed_text = (
                    f"seed {seeds[0]}" if len(seeds) == 1
                    else "seeds " + ", ".join(map(str, seeds))
                )
                title = (
                    f"{labels.get(experiment, experiment)} · trial {trial} · "
                    f"episode {episode} ({seed_text}) · deterministic policy"
                )
                out_path = args.out_dir / f"{experiment}_trial{trial}_episode{episode}.png"
                plot_episode_grid(
                    batches, panels, title, max_rewards, args.columns, out_path, args.dpi
                )
                written.append(out_path)
                print(out_path)

    if missing:
        print(f"{len(missing)} panels without a figure (empty panel):")
        for item in missing:
            print(f"  {item}")
    if stale:
        print(f"{len(stale)} figures older than their run's saved model (flagged):")
        for item in stale:
            print(f"  {item}")
    if not written:
        raise SystemExit(
            "no trajectory figures found; render them first with "
            "scripts/view_results.sh"
        )
    print(f"{len(written)} figures in {args.out_dir}")


if __name__ == "__main__":
    main()
