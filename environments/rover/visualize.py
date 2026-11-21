"""Matplotlib figures for the rover domain: arena layouts and example trajectories.

* ``plot_arena(env)`` — the arena and its POIs. A POI is drawn as its
  observation-radius disc with its coupling (rovers needed at once) written
  inside, plus its value and per-step rate (and its consumption order, when
  the layout has more than one). Rover start positions are random,
  so they are not drawn; one rover's sensing radius is overlaid at the
  spawn-region centre (``draw_sensing_radius``).
* ``plot_trajectory(env, traj)`` — one episode: every rover's path (start dot,
  arrow at the final heading) over the arena, when each POI paid, and the
  cumulative team reward. Used-up POIs are drawn hollow with their depletion
  step.
* ``plot_poi_masks(env, traj)`` — what the ``rmappo_jax`` POI manager showed
  over one ``rollout_view_policy`` episode: each POI's disc filled by the share
  of steps it was shown (a sequential heatmap, no rovers drawn), and per-step
  rows of when it was shown and when it paid. ``view=true`` writes it for a
  run with the manager.

CLI (writes one arena image per ``conf/env/rover_*.yaml`` variant, plus
trajectory figures for the random and scripted demonstration policies)::

    uv run python -m environments.rover.visualize \
        [--variants rover_2poi_c2 ...] [--seeds 0 1] \
        [--policies random scripted] [--out-dir plotting/rover] [--no-trajectories]

Outputs: ``<out-dir>/arenas/<variant>.png`` and
``<out-dir>/trajectories/<variant>_<policy>_seed<k>.png``.

For trained policies, use the runner's view command instead::

    uv run python train.py algorithm=rmappo_jax env=rover_2poi_c2 \
        model=rmappo trial_id=0 view=true

``model=mappo_mlp`` selects the feedforward policy. The runner writes to
``experiments/results/<env>/<model>/<trial>/logs/rover/``.
"""

import argparse
from pathlib import Path

import jax
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.cm import ScalarMappable
from matplotlib.colors import LinearSegmentedColormap, LogNorm, Normalize, to_rgba
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, Rectangle

from environments.rover.config import REPO_ROOT, list_variants, load_rover_env
from environments.rover.policies import make_scripted_coalition_policy, random_policy
from environments.rover.rollout import rollout

# Light-mode chart chrome and the categorical order of the dataviz reference
# palette (fixed order, never cycled; validated for line charts).
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
# The reference palette's sequential blue ramp, steps 100, 150, ..., 700 (one
# hue, light = near zero, dark = most), for continuous magnitudes such as the
# share of steps the POI manager showed a POI. Index 7 is step 450 (#2a78d6).
SEQUENTIAL_BLUE = [
    "#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7", "#3987e5",
    "#2a78d6", "#256abf", "#1c5cab", "#184f95", "#104281", "#0d366b",
]

# The reference palette's diverging pair, blue <-> red around its light grey
# midpoint (the slot-1 blue and slot-8 red, each with a darker end), for a
# signed error such as the critic's V - G.
DIVERGING = ["#0d366b", "#2a78d6", "#f0efec", "#e34948", "#8f2523"]
# A second sequential context takes the next categorical hue (orange) as its
# own one-hue ramp, light = few, dark = many: the visits per cell, kept apart
# from the blue of the values.
SEQUENTIAL_ORANGE = ["#fbe3d6", "#f4a37f", "#eb6834", "#b8461a", "#7a2c0e"]

POLICIES = {
    "random": lambda env: random_policy,
    "scripted": make_scripted_coalition_policy,
}


def _style_axes(ax):
    ax.set_facecolor(SURFACE)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(colors=MUTED, labelsize=8, length=0)


def _is_ordered(pois):
    """True when the layout uses temporal coupling (more than one order)."""
    return len(np.unique(np.asarray(pois.order))) > 1


def draw_arena(ax, env, pois=None, depleted_at=None, coupling_inside=True, compact=False):
    """Draw the arena and POIs on ``ax``.

    ``pois`` defaults to the env's initial POIs (reset key 0). ``depleted_at``
    maps a POI index to the step it was used up; such POIs are drawn hollow.
    ``coupling_inside`` writes the coupling large in the disc centre; set it
    False when paths will be drawn on top (they end on the centre), which moves
    the coupling into the label above the disc. A layout with more than one
    consumption order adds each POI's order to that label.

    ``compact`` is for small panels over a heatmap (the critic value maps):
    every disc is an outline (a used-up one dashed), so the heatmap shows
    through, and each POI gets only its ``P<k>`` label.
    """
    if pois is None:
        pois = env.pois.init(jax.random.PRNGKey(0), env)
    depleted_at = depleted_at or {}
    ordered = _is_ordered(pois)
    width, height = env.world_size
    _style_axes(ax)
    ax.set_aspect("equal")
    pad = 0.04 * max(width, height)
    ax.set_xlim(-pad, width + pad)
    ax.set_ylim(-pad, height + pad)
    ticks = max(1, round(max(width, height) / 5))
    ax.set_xticks(np.arange(0, width + 1e-9, ticks))
    ax.set_yticks(np.arange(0, height + 1e-9, ticks))
    ax.grid(True, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    ax.add_patch(Rectangle((0, 0), width, height, fill=False, edgecolor=AXIS, linewidth=1.2))

    position = np.asarray(pois.position)
    for k in range(position.shape[0]):
        x, y = position[k]
        radius = float(pois.radius[k])
        used = k in depleted_at
        if compact:
            ax.add_patch(Circle((x, y), radius, facecolor="none", edgecolor=INK,
                                linewidth=1.0, linestyle="--" if used else "-",
                                zorder=3))
            ax.text(x, y + radius + 0.08, f"P{k}", ha="center", va="bottom",
                    fontsize=7, color=INK, zorder=4,
                    bbox=dict(boxstyle="round,pad=0.1", facecolor=SURFACE,
                              edgecolor="none", alpha=0.8))
            continue
        ax.add_patch(
            Circle(
                (x, y),
                radius,
                facecolor="none" if used else to_rgba(INK_2, 0.10),
                edgecolor=MUTED if used else INK_2,
                linewidth=1.2,
                zorder=2,
            )
        )
        coupling = int(pois.coupling[k])
        if coupling_inside:
            ax.text(x, y, str(coupling), ha="center", va="center", fontsize=13,
                    fontweight="bold", color=MUTED if used else INK, zorder=4)
            label = f"P{k}"
        else:
            ax.plot(x, y, "+", color=MUTED if used else INK_2, markersize=7, zorder=2)
            label = f"P{k} · c={coupling}"
        if ordered:
            label += f" · order {int(pois.order[k])}"
        ax.text(x, y + radius + 0.12, label, ha="center", va="bottom",
                fontsize=8.5, color=INK_2, zorder=4)
        detail = (
            f"used at t={depleted_at[k]}"
            if used
            else f"v={float(pois.value[k]):g}, {float(pois.rate[k]):g}/step"
        )
        ax.text(x, y - radius - 0.12, detail, ha="center", va="top",
                fontsize=7.5, color=MUTED, zorder=4)


def draw_sensing_radius(ax, env, pois=None):
    """Overlay one rover's sensing radius, centred on the spawn region's centre.

    Rover start positions are random, so the disc shows the sensing range at a
    representative start rather than at any particular rover. Does nothing for
    an observation model without a ``radius(env)``. When the radius covers the
    whole arena from that point (the ``sensor_radius: null`` default, R = the
    arena diagonal) the circle lies outside the plot, so a label says so instead.
    """
    radius_fn = getattr(env.observation, "radius", None)
    if radius_fn is None:
        return
    if pois is None:
        pois = env.pois.init(jax.random.PRNGKey(0), env)
    radius = float(radius_fn(env))
    x_min, x_max, y_min, y_max = env.spawn_region
    centre = np.array([(x_min + x_max) / 2, (y_min + y_max) / 2])
    width, height = env.world_size
    corners = np.array([[0, 0], [width, 0], [0, height], [width, height]])
    covers_arena = radius >= np.max(np.linalg.norm(corners - centre, axis=1))
    colour = SERIES[0]

    ax.plot(*centre, "o", color=colour, markersize=6, markeredgecolor=SURFACE,
            markeredgewidth=1.5, zorder=5)
    if covers_arena:
        ax.text(centre[0], centre[1] - 0.25,
                f"sensing radius {radius:.3g}\n(senses the whole arena)",
                ha="center", va="top", fontsize=7.5, color=colour, zorder=5)
        return

    ax.add_patch(Circle(centre, radius, facecolor=to_rgba(colour, 0.06),
                        edgecolor=colour, linewidth=1.2, linestyle="--", zorder=3))
    # Draw the radius along the direction that stays clearest of the POI discs.
    position = np.asarray(pois.position)
    poi_radius = np.asarray(pois.radius)
    best_angle, best_clearance = 0.0, -np.inf
    for angle in np.arange(8) * np.pi / 4 + np.pi / 4:
        tip = centre + radius * np.array([np.cos(angle), np.sin(angle)])
        samples = centre + np.linspace(0, 1, 9)[:, None] * (tip - centre)
        gaps = np.linalg.norm(samples[:, None] - position[None], axis=-1) - poi_radius
        clearance = gaps.min() if gaps.size else np.inf
        if clearance > best_clearance:
            best_angle, best_clearance = angle, clearance
    direction = np.array([np.cos(best_angle), np.sin(best_angle)])
    tip = centre + radius * direction
    ax.plot(*zip(centre, tip), color=colour, linewidth=1.0, zorder=4)
    middle = centre + 0.5 * radius * direction
    # Anchor the label on the side the line's normal points to, so it never
    # covers the line whatever the angle.
    normal = np.array([-direction[1], direction[0]])
    ha = "left" if normal[0] > 0.3 else "right" if normal[0] < -0.3 else "center"
    va = "bottom" if normal[1] > 0.3 else "top" if normal[1] < -0.3 else "center"
    ax.text(*(middle + 0.12 * normal), f"sensing R = {radius:g}", ha=ha, va=va,
            fontsize=7.5, color=colour, zorder=5,
            bbox=dict(boxstyle="round,pad=0.15", facecolor=SURFACE, edgecolor="none"))


def plot_arena(env, title=""):
    fig, ax = plt.subplots(figsize=(6.2, 6.6), facecolor=SURFACE)
    draw_arena(ax, env)
    draw_sensing_radius(ax, env)
    width, height = env.world_size
    pois = env.pois.init(jax.random.PRNGKey(0), env)
    total = float(np.sum(pois.value))
    ax.set_title(
        f"{title}\n{env.n_agents} rovers · {env.n_pois} POIs · {env.max_steps} steps · "
        f"{width:g} x {height:g} arena · max return {total:g}",
        fontsize=10, color=INK, loc="left",
    )
    note = (
        "Number = coupling (rovers needed inside the disc at once). "
        "Grey disc = POI observation radius.\n"
        "Blue = one rover's sensing radius, drawn at the spawn-region centre."
    )
    if _is_ordered(pois):
        note += "\nOrder n = the POI pays only once every POI of a lower order is used up."
    fig.text(0.02, 0.01, note, fontsize=7.5, color=MUTED)
    return fig


def _runs(mask):
    """(start, length) of every run of True in a 1-D bool array."""
    padded = np.concatenate([[False], mask, [False]]).astype(int)
    edges = np.flatnonzero(np.diff(padded))
    return [(s, e - s) for s, e in zip(edges[::2], edges[1::2])]


def _episode_length(done):
    """Steps the episode lasted: the first step after which ``done`` (T,) is
    set, or T when it never is."""
    done = np.asarray(done)
    return int(np.argmax(done)) + 1 if done.any() else done.shape[0]


def _depleted_at(remaining):
    """POI index -> the step it was used up, for each POI used up in the
    episode. ``remaining`` (T+1, P) are the budgets, index 0 = reset."""
    return {
        k: int(np.argmax(remaining[1:, k] <= 0)) + 1
        for k in range(remaining.shape[1])
        if np.any(remaining[1:, k] <= 0)
    }


def _poi_rows(ax, active, pois, n_steps, colour, title, idle_label):
    """One row per POI (POI 0 on top) over ``n_steps`` steps, filled where
    ``active`` (T, P) is true; ``idle_label`` is written in a row never active.
    The row of POI k sits at y = P - 1 - k."""
    _style_axes(ax)
    n_pois = active.shape[1]
    for k in range(n_pois):
        row = n_pois - 1 - k
        ax.broken_barh(_runs(active[:, k]), (row - 0.3, 0.6),
                       facecolors=colour, edgecolor=SURFACE, linewidth=0)
        if not np.any(active[:, k]):
            ax.text(0.01 * n_steps, row, idle_label, va="center",
                    fontsize=7.5, color=MUTED)
    ax.set_yticks(range(n_pois))
    ordered = _is_ordered(pois)
    ax.set_yticklabels([
        f"P{k} (c={int(pois.coupling[k])}"
        + (f", order {int(pois.order[k])})" if ordered else ")")
        for k in reversed(range(n_pois))
    ])
    ax.set_xlim(0, n_steps)
    ax.set_ylim(-0.7, n_pois - 0.3)
    ax.grid(True, axis="x", color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    ax.set_title(title, fontsize=10, color=INK, loc="left")


def _mark_used(ax, depleted_at, n_pois, n_steps):
    """Write "used" in each used-up POI's row of a ``_poi_rows`` panel, just
    after the step it was used up."""
    for k, step in depleted_at.items():
        ax.text(step + 0.01 * n_steps, n_pois - 1 - k, "used", va="center",
                fontsize=7.5, color=MUTED)


def plot_trajectory(env, traj, title=""):
    """Figure of one ``rollout`` episode (an unbatched ``Trajectory``)."""
    positions = np.asarray(traj.positions)
    headings = np.asarray(traj.headings)
    remaining = np.asarray(traj.poi_remaining)
    rewards = np.asarray(traj.rewards)
    if rewards.ndim == 2:  # per-agent rewards
        rewards = rewards.sum(axis=1)
    n_steps, n_agents = rewards.shape[0], positions.shape[1]
    length = _episode_length(traj.done)

    payout = remaining[:-1] - remaining[1:]  # (T, P)
    depleted_at = _depleted_at(remaining)
    pois = env.pois.init(jax.random.PRNGKey(0), env)

    fig = plt.figure(figsize=(13, 6.4), facecolor=SURFACE)
    grid = GridSpec(2, 2, figure=fig, width_ratios=[1.0, 1.1], hspace=0.45, wspace=0.18)
    ax_arena = fig.add_subplot(grid[:, 0])
    draw_arena(ax_arena, env, pois, depleted_at, coupling_inside=False)

    # More rovers than categorical slots fold into one neutral colour.
    colours = SERIES if n_agents <= len(SERIES) else [INK_2] * n_agents
    arrow = 0.04 * max(env.world_size)
    for i in range(n_agents):
        c = colours[i]
        path = positions[: length + 1, i]
        ax_arena.plot(path[:, 0], path[:, 1], color=c, linewidth=1.5,
                      solid_capstyle="round", solid_joinstyle="round", zorder=3)
        ax_arena.plot(*path[0], "o", color=c, markersize=6,
                      markeredgecolor=SURFACE, markeredgewidth=1.5, zorder=5)
        end, h = path[-1], headings[length, i]
        ax_arena.annotate(
            "", xy=(end[0] + arrow * np.cos(h), end[1] + arrow * np.sin(h)), xytext=end,
            arrowprops=dict(arrowstyle="-|>", color=c, linewidth=1.5, mutation_scale=12),
            zorder=5,
        )
    if n_agents <= len(SERIES):
        handles = [Line2D([], [], color=colours[i], linewidth=2, label=f"R{i}")
                   for i in range(n_agents)]
    else:
        handles = [Line2D([], [], color=INK_2, linewidth=2, label=f"{n_agents} rovers")]
    handles.append(Line2D([], [], linestyle="none", marker="o", color=INK_2,
                          markersize=6, label="start"))
    ax_arena.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.05),
                    ncol=min(len(handles), 7), frameon=False, fontsize=8,
                    labelcolor=INK_2, handlelength=1.6)

    # When each POI paid: one row per POI, filled where its coupling was met.
    ax_pay = fig.add_subplot(grid[0, 1])
    _poi_rows(ax_pay, payout > 0, pois, n_steps, INK_2,
              "Steps where each POI paid (coupling met)", "never paid")
    _mark_used(ax_pay, depleted_at, remaining.shape[1], n_steps)

    # Cumulative team reward, against the most the episode could pay.
    ax_ret = fig.add_subplot(grid[1, 1], sharex=ax_pay)
    _style_axes(ax_ret)
    total = float(np.sum(pois.value))
    cumulative = np.concatenate([[0.0], np.cumsum(rewards)])[: length + 1]
    ax_ret.plot(np.arange(length + 1), cumulative, color=INK, linewidth=1.5)
    ax_ret.axhline(total, color=AXIS, linewidth=1.0)
    ax_ret.text(0.01 * n_steps, total, f"every POI used = {total:g}", ha="left",
                va="bottom", fontsize=7.5, color=MUTED)
    if length < n_steps:
        for ax in (ax_pay, ax_ret):
            ax.axvline(length, color=AXIS, linewidth=1.0)
        # Right of the line when there is room: the curve rises left of it.
        right = length < 0.7 * n_steps
        ax_ret.text(length, 0.02 * total,
                    f" episode ends t={length}" if right else f"episode ends t={length} ",
                    ha="left" if right else "right", va="bottom", fontsize=7.5, color=MUTED)
    ax_ret.set_ylim(0, 1.12 * total)
    ax_ret.grid(True, axis="y", color=GRID, linewidth=0.6)
    ax_ret.set_axisbelow(True)
    ax_ret.set_xlabel("step", fontsize=8, color=MUTED)
    ax_ret.set_title("Cumulative team reward", fontsize=10, color=INK, loc="left")

    fig.suptitle(
        f"{title}   return {float(np.sum(rewards)):.2f} of {total:g} in {length} steps",
        x=0.02, ha="left", fontsize=11, color=INK,
    )
    return fig


def poi_shown_share(traj):
    """(P,) share of the episode's steps on which the POI manager showed each
    POI (``Trajectory.poi_masks``), in [0, 1]."""
    if traj.poi_masks is None:
        raise ValueError("the trajectory has no POI masks (no POI manager)")
    masks = np.asarray(traj.poi_masks, dtype=bool)
    return masks[: _episode_length(traj.done)].mean(axis=0)


def plot_poi_masks(env, traj, title=""):
    """What the POI manager showed over one ``rollout_view_policy`` episode.

    Left: the arena (no rovers), each POI's disc filled by the share of the
    episode's steps on which the manager's mask showed it (``poi_shown_share``,
    sequential blue, fixed 0-100%), the share written inside. Right, one row
    per POI on a shared step axis: the steps the manager showed it, and the
    steps it paid. Shown is the manager's own mask; a POI the env itself hides
    (``pois.visible``) is not sensed even when shown.
    """
    share = poi_shown_share(traj)
    masks = np.asarray(traj.poi_masks, dtype=bool)
    remaining = np.asarray(traj.poi_remaining)
    rewards = np.asarray(traj.rewards)
    if rewards.ndim == 2:  # per-agent rewards
        rewards = rewards.sum(axis=1)
    length = _episode_length(traj.done)
    n_steps, n_pois = env.max_steps, remaining.shape[1]
    depleted_at = _depleted_at(remaining)
    pois = env.pois.init(jax.random.PRNGKey(0), env)

    fig = plt.figure(figsize=(13, 6.4), facecolor=SURFACE)
    grid = GridSpec(2, 2, figure=fig, width_ratios=[1.0, 1.1], hspace=0.45, wspace=0.18)
    ax_arena = fig.add_subplot(grid[:, 0])
    draw_arena(ax_arena, env, pois, depleted_at, coupling_inside=False)
    cmap = LinearSegmentedColormap.from_list("poi_shown", SEQUENTIAL_BLUE)
    norm = Normalize(0.0, 1.0)
    position = np.asarray(pois.position)
    for k in range(n_pois):
        x, y = position[k]
        # Above draw_arena's disc (zorder 2), below its labels (zorder 4).
        ax_arena.add_patch(Circle((x, y), float(pois.radius[k]),
                                  facecolor=cmap(norm(share[k])), edgecolor=INK_2,
                                  linewidth=1.2, zorder=3))
        ax_arena.text(x, y, f"{share[k]:.0%}", ha="center", va="center",
                      fontsize=8.5, color=INK, zorder=5,
                      bbox=dict(boxstyle="round,pad=0.15", facecolor=SURFACE,
                                edgecolor="none"))
    bar = fig.colorbar(ScalarMappable(norm=norm, cmap=cmap), ax=ax_arena,
                       orientation="horizontal", fraction=0.04, pad=0.06)
    bar.outline.set_visible(False)
    bar.set_ticks([0, 0.25, 0.5, 0.75, 1.0])
    bar.set_ticklabels(["0%", "25%", "50%", "75%", "100%"])
    bar.ax.tick_params(colors=MUTED, labelsize=8, length=0)
    bar.set_label("share of the episode's steps the manager showed the POI",
                  fontsize=8, color=INK_2)

    # One row per POI: where the manager showed it, then where it paid.
    ax_shown = fig.add_subplot(grid[0, 1])
    _poi_rows(ax_shown, masks, pois, n_steps, SEQUENTIAL_BLUE[7],
              "Steps where the manager showed each POI (| = used up)", "never shown")
    for k, step in depleted_at.items():
        row = n_pois - 1 - k
        ax_shown.plot([step, step], [row - 0.42, row + 0.42], color=INK,
                      linewidth=1.5, solid_capstyle="butt", zorder=4)
    ax_pay = fig.add_subplot(grid[1, 1], sharex=ax_shown)
    payout = remaining[:-1] - remaining[1:]  # (T, P)
    _poi_rows(ax_pay, payout > 0, pois, n_steps, INK_2,
              "Steps where each POI paid (coupling met)", "never paid")
    _mark_used(ax_pay, depleted_at, n_pois, n_steps)
    ax_pay.set_xlabel("step", fontsize=8, color=MUTED)
    if length < n_steps:
        for ax in (ax_shown, ax_pay):
            ax.axvline(length, color=AXIS, linewidth=1.0)
        ax_pay.text(length, -0.65, f" episode ends t={length}", ha="left",
                    va="bottom", fontsize=7.5, color=MUTED)

    total = float(np.sum(pois.value))
    fig.suptitle(
        f"{title}   return {float(np.sum(rewards)):.2f} of {total:g} in {length} steps",
        x=0.02, ha="left", fontsize=11, color=INK,
    )
    fig.text(
        0.02, 0.01,
        "Shown = the POI manager's own mask (deterministic: shown when its logit "
        "is >= 0). A POI the environment hides (locked or used up, in _hidden "
        "groups) is not sensed even when shown.",
        fontsize=7.5, color=MUTED,
    )
    return fig


def _aspect(env):
    width, height = env.world_size
    return height / width


def _panel_width(fig_width, n_panels, wspace=0.12):
    """Inches each of ``n_panels`` side-by-side GridSpec axes gets in a
    figure ``fig_width`` wide (matplotlib's default 0.125 / 0.9 side margins),
    so a figure can be made just tall enough for equal-aspect arenas."""
    return 0.775 * fig_width / (n_panels + wspace * (n_panels - 1))


def _spread_labels(ys, gap):
    """End-label heights for ``ys``, pushed apart to at least ``gap`` so two
    lines that end together keep readable labels; the order is kept."""
    order = np.argsort(ys)
    placed = np.asarray(ys, np.float64)[order].copy()
    for i in range(1, len(placed)):
        placed[i] = max(placed[i], placed[i - 1] + gap)
    placed -= (placed.mean() - np.mean(ys)) if len(placed) > 1 and np.ptp(ys) < gap else 0.0
    out = np.empty_like(placed)
    out[order] = placed
    return out


def _colormap(name, colours):
    """A colormap from a palette ramp, NaN drawn transparent (blank cells show
    the arena underneath)."""
    return LinearSegmentedColormap.from_list(name, colours).with_extremes(bad=(0, 0, 0, 0))


def _heatmap(ax, env, grid, cmap, norm):
    """``grid (ny, nx)`` over the whole arena, row 0 at the bottom, above the
    arena's grid lines (zorder 0.5) and below its POI outlines (zorder 3)."""
    width, height = env.world_size
    return ax.imshow(grid, extent=(0, width, 0, height), origin="lower", cmap=cmap,
                     norm=norm, interpolation="nearest", zorder=1)


def _colourbar(fig, axes, cmap, norm, label, orientation="vertical", **kwargs):
    bar = fig.colorbar(ScalarMappable(norm=norm, cmap=cmap), ax=axes,
                       orientation=orientation, **kwargs)
    bar.outline.set_visible(False)
    bar.ax.tick_params(colors=MUTED, labelsize=7.5, length=0)
    bar.set_label(label, fontsize=8, color=INK_2)
    return bar


def plot_value_slices(env, traj, phases, slices, xs, ys, values, returns, *,
                      gamma, n_headings, recurrent, title=""):
    """The critic's value over the arena at a few steps of one episode.

    One panel per phase (``value_map.phase_steps``): the critic's value with
    the whole team translated so its centre sits on each cell
    (``CriticProbe.team_slice``), every panel on one colour scale, with the
    POIs (used up by then dashed), the rovers at that step and the team
    centre's path so far. Below, along the real episode: the critic's value
    ``values`` against the realized discounted return-to-go ``returns``, both
    ``(T+1,)``, over the steps the critic is queried at (``0 .. T-1``).
    """
    positions = np.asarray(traj.positions)
    remaining = np.asarray(traj.poi_remaining)
    n_steps = int(np.asarray(traj.rewards).shape[0])
    n_agents = positions.shape[1]
    centre = positions.mean(axis=1)  # (T+1, 2)
    n_panels = len(phases)
    stacked = np.stack(slices)
    norm = Normalize(min(0.0, float(np.nanmin(stacked))), float(np.nanmax(stacked)))
    cmap = _colormap("critic_value", SEQUENTIAL_BLUE)
    colours = SERIES if n_agents <= len(SERIES) else [INK_2] * n_agents

    fig_width = max(13.0, 3.3 * n_panels + 1.5)
    panel_height = _panel_width(fig_width, n_panels) * _aspect(env)
    strip_height = 1.45
    fig = plt.figure(figsize=(fig_width, panel_height + strip_height + 2.2),
                     facecolor=SURFACE)
    grid = GridSpec(2, n_panels, figure=fig, height_ratios=[panel_height, strip_height],
                    hspace=0.42, wspace=0.12, top=0.86)
    axes = []
    for i, ((t, label), value_grid) in enumerate(zip(phases, slices)):
        ax = fig.add_subplot(grid[0, i])
        axes.append(ax)
        _heatmap(ax, env, value_grid, cmap, norm)
        used = _depleted_at(remaining[: t + 1]) if t > 0 else {}
        draw_arena(ax, env, jax.tree.map(lambda x: x[t], traj.states.pois), used,
                   compact=True)
        ax.plot(centre[: t + 1, 0], centre[: t + 1, 1], color=INK, linewidth=1.0,
                alpha=0.6, solid_capstyle="round", zorder=4)
        for r in range(n_agents):
            ax.plot(*positions[t, r], "o", color=colours[r], markersize=5,
                    markeredgecolor=SURFACE, markeredgewidth=1.2, zorder=5)
        if i:
            ax.tick_params(labelleft=False)
        heading = f"t = {t}" + (f" · {label}" if label else "")
        ax.set_title(
            f"{heading}\nV {values[t]:.2f} · realized G {returns[t]:.2f}\n"
            f"map {np.nanmin(value_grid):.2f} to {np.nanmax(value_grid):.2f}",
            fontsize=9, color=INK, loc="left",
        )
    _colourbar(fig, axes, cmap, norm,
               "critic value V (discounted return, reward units)",
               fraction=0.012, pad=0.01)

    # Along the real path: what the critic predicted against what came.
    ax = fig.add_subplot(grid[1, :])
    _style_axes(ax)
    steps = np.arange(n_steps)
    ax.plot(steps, values[:n_steps], color=SERIES[0], linewidth=2, label="critic value V")
    ax.plot(steps, returns[:n_steps], color=SERIES[1], linewidth=2,
            label="realized discounted return-to-go G")
    ends = _spread_labels([values[n_steps - 1], returns[n_steps - 1]],
                          gap=0.08 * max(np.ptp(np.concatenate([values[:n_steps],
                                                                returns[:n_steps]])), 1e-6))
    for name, y in zip(("V", "G"), ends):
        ax.text(n_steps - 1 + 0.01 * n_steps, y, name, va="center", fontsize=8,
                color=INK_2)
    for t, _ in phases:
        ax.axvline(t, color=AXIS, linewidth=1.0, zorder=0)
    ax.set_xlim(0, max(n_steps - 1, 1) * 1.04)
    ax.grid(True, axis="y", color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    ax.set_xlabel("step (vertical lines: the panels above)", fontsize=8, color=MUTED)
    ax.set_title("Along the episode", fontsize=10, color=INK, loc="left")
    ax.legend(loc="upper right", frameon=False, fontsize=8, labelcolor=INK_2, ncol=2)

    team = "the rover" if n_agents == 1 else "the whole team, moved as a block,"
    memory = ", and the critic's GRU memory as carried into that step" if recurrent else ""
    heading = (
        f"Mean over {n_headings} headings, the team turning as a block."
        if n_headings > 1
        else "The critic reads the global state, which holds no heading."
    )
    note = (
        f"Each panel: the critic's value with {team} placed so its centre sits on "
        f"each cell; the formation and the POI budgets are as at that step{memory}. "
        f"{heading}\nV = the critic's prediction of the discounted team return "
        f"(gamma = {gamma:g}); G = what this episode then paid, discounted the same "
        "way. Away from the path the map is extrapolation: the critic trained only "
        "on visited states."
    )
    if recurrent:
        note += (" Its memory still holds the real history, so a far cell is a "
                 "teleported team.")
    fig.text(0.02, 0.005, note, fontsize=7.5, color=MUTED, va="bottom")
    fig.suptitle(f"{title}   critic value maps", x=0.02, ha="left", fontsize=11,
                 color=INK)
    return fig


def plot_value_visitation(env, maps, n_episodes, n_states, *, gamma, title=""):
    """The critic's value where episodes actually go (``value_map.visitation_maps``).

    Four arena panels binned by the team centre: the mean critic value, the
    mean realized discounted return-to-go (same scale), their difference (the
    critic's error, diverging around 0) and the visits per cell (log scale).
    Cells with fewer than ``maps.min_visits`` states are blank.
    """
    value_cmap = _colormap("critic_value", SEQUENTIAL_BLUE)
    both = np.concatenate([maps.mean_value.ravel(), maps.mean_return.ravel()])
    if np.all(np.isnan(both)):
        both = np.zeros(1)
    value_norm = Normalize(min(0.0, float(np.nanmin(both))), float(np.nanmax(both)))
    error = maps.mean_value - maps.mean_return
    spread = float(np.nanmax(np.abs(error))) if np.any(np.isfinite(error)) else 1.0
    error_norm = Normalize(-max(spread, 1e-6), max(spread, 1e-6))
    visits = np.where(maps.visits > 0, maps.visits, np.nan)
    visit_norm = LogNorm(1.0, max(2.0, float(np.nanmax(maps.visits))))

    panels = [
        (maps.mean_value, value_cmap, value_norm, "Critic value V",
         "mean critic value V"),
        (maps.mean_return, value_cmap, value_norm, "Realized return-to-go G",
         "mean realized discounted return-to-go G"),
        (error, _colormap("critic_error", DIVERGING), error_norm,
         "Critic error V - G", "V - G (red: critic too high, blue: too low)"),
        (visits, _colormap("visits", SEQUENTIAL_ORANGE), visit_norm,
         "Visits", "visited states per cell (log scale)"),
    ]
    fig_width = 16.5
    fig = plt.figure(figsize=(fig_width, _panel_width(fig_width, 4) * _aspect(env) + 1.5),
                     facecolor=SURFACE)
    grid = GridSpec(1, 4, figure=fig, wspace=0.12, top=0.86, bottom=0.2)
    for i, (data, cmap, norm, name, label) in enumerate(panels):
        ax = fig.add_subplot(grid[0, i])
        _heatmap(ax, env, data, cmap, norm)
        draw_arena(ax, env, compact=True)
        if i:
            ax.tick_params(labelleft=False)
        ax.set_title(name, fontsize=10, color=INK, loc="left")
        _colourbar(fig, ax, cmap, norm, label, orientation="horizontal",
                   fraction=0.05, pad=0.08)

    fig.text(
        0.02, 0.005,
        f"{n_episodes} episodes with sampled actions, as in training ({n_states} "
        "states). Each state is binned by the team centre; cells with fewer than "
        f"{maps.min_visits} states are blank. V = the critic's value at the visited "
        "state, with its true memory; G = the discounted team return that followed "
        f"(gamma = {gamma:g}).\nThe values mix every phase of the episode, so read "
        "them together with the per-episode value maps. POIs are drawn at their "
        "start; none is shown used up.",
        fontsize=7.5, color=MUTED, va="bottom",
    )
    fig.suptitle(f"{title}   critic value where the team goes", x=0.02, ha="left",
                 fontsize=11, color=INK)
    return fig


def save_figure(fig, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)
    return path


def main(argv=None):
    matplotlib.use("Agg")
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--variants", nargs="*", default=None,
                        help="rover env groups (default: every conf/env/rover_*.yaml)")
    parser.add_argument("--seeds", nargs="+", type=int, default=[0])
    parser.add_argument("--policies", nargs="+", choices=sorted(POLICIES),
                        default=["random", "scripted"])
    parser.add_argument("--out-dir", type=Path, default=REPO_ROOT / "plotting" / "rover")
    parser.add_argument("--no-trajectories", action="store_true",
                        help="only write the arena images")
    args = parser.parse_args(argv)

    for variant in args.variants or list_variants():
        env = load_rover_env(variant)
        print(save_figure(plot_arena(env, variant), args.out_dir / "arenas" / f"{variant}.png"))
        if args.no_trajectories:
            continue
        for name in args.policies:
            policy = POLICIES[name](env)
            run = jax.jit(lambda key: rollout(env, policy, key))
            for seed in args.seeds:
                traj = run(jax.random.PRNGKey(seed))
                fig = plot_trajectory(env, traj, f"{variant} · {name} policy · seed {seed}")
                out = args.out_dir / "trajectories" / f"{variant}_{name}_seed{seed}.png"
                print(save_figure(fig, out))


if __name__ == "__main__":
    main()
