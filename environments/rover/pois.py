"""POI models. ``StaticPOIs`` is the fixed, config-defined layout.

Every POI carries a consumption ``order`` (its temporal coupling, next to the
spatial ``coupling``): it can pay only once every POI of a lower order is used
up. ``unlocked`` is the one definition of that rule; the reward and the
scripted policies both read it. A POI flagged ``hide_until_unlocked`` is also
hidden from the rovers while it is locked; ``visible`` is the one definition of
what the rovers can sense.
"""

import warnings

import jax.numpy as jnp
import numpy as np

from environments.rover.components import POIModel
from environments.rover.state import POIState

_ENTRY_KEYS = {
    "position",
    "coupling",
    "radius",
    "value",
    "rate",
    "order",
    "hide_until_unlocked",
}


def unlocked(pois):
    """(..., P) bool: POIs whose temporal coupling is met, i.e. every POI of a
    lower ``order`` is used up. Says nothing about a POI's own budget.

    The lowest order among the POIs with budget left is always unlocked, so a
    POI with budget left is always available somewhere.
    """
    pending = pois.remaining > 0  # (..., P)
    earlier = pois.order[..., None, :] < pois.order[..., :, None]  # [p, q]: q before p
    return ~jnp.any(earlier & pending[..., None, :], axis=-1)


def visible(pois):
    """(..., P) bool: POIs the rovers can sense. A used-up POI never is, and a
    POI flagged ``hide_until_unlocked`` is not while it is locked (``unlocked``).
    """
    return (pois.remaining > 0) & (~pois.hide_until_unlocked | unlocked(pois))


class StaticPOIs(POIModel):
    """A fixed POI layout, identical in every episode.

    Args:
        entries: one mapping per POI with ``position: [x, y]`` and ``coupling``
            (required) and optional ``radius`` / ``value`` / ``rate``, which
            override the defaults below for that POI, and ``order`` (an
            integer >= 0, default 0): the POI pays only once every POI of a
            lower order is used up (see ``unlocked``). With every order equal,
            as by default, POIs can be used in any order. An entry may also set
            ``hide_until_unlocked`` (overriding the default below).
        radius: observation radius (inclusive) a rover must be within.
        value: total reward the POI can pay in one episode.
        rate: reward paid per step while its coupling is met; the POI is used
            up after ``ceil(value / rate)`` coupled steps.
        hide_until_unlocked: true or false. A flagged POI is not sensed by the
            rovers while it is locked, i.e. while a POI of a lower ``order``
            still has budget (see ``visible``). A POI of the lowest order is
            never locked, so the flag only acts on a POI with a higher order.
    """

    def __init__(self, entries, radius=1.0, value=1.0, rate=0.05, hide_until_unlocked=False):
        entries = [dict(e) for e in entries]
        if not entries:
            raise ValueError("StaticPOIs needs at least one POI entry")
        rows = []
        for k, entry in enumerate(entries):
            unknown = set(entry) - _ENTRY_KEYS
            if unknown:
                raise ValueError(
                    f"POI {k}: unknown keys {sorted(unknown)}; "
                    f"allowed {sorted(_ENTRY_KEYS)}"
                )
            missing = {"position", "coupling"} - set(entry)
            if missing:
                raise ValueError(f"POI {k}: missing required keys {sorted(missing)}")
            coupling = entry["coupling"]
            if int(coupling) != coupling or int(coupling) < 1:
                raise ValueError(
                    f"POI {k}: coupling must be an integer >= 1, got {coupling}"
                )
            order = entry.get("order", 0)
            if int(order) != order or int(order) < 0:
                raise ValueError(f"POI {k}: order must be an integer >= 0, got {order}")
            hide = entry.get("hide_until_unlocked", hide_until_unlocked)
            if not isinstance(hide, (bool, np.bool_)):
                raise ValueError(
                    f"POI {k}: hide_until_unlocked must be true or false, got {hide!r}"
                )
            rows.append(
                (
                    entry["position"],
                    int(coupling),
                    float(entry.get("radius", radius)),
                    float(entry.get("value", value)),
                    float(entry.get("rate", rate)),
                    int(order),
                    bool(hide),
                )
            )
        position, coupling, radius_, value_, rate_, order_, hide_ = zip(*rows)
        self.position = np.asarray(position, dtype=np.float32)
        if self.position.shape != (len(rows), 2):
            raise ValueError(f"every POI position must be [x, y], got {position}")
        self.coupling = np.asarray(coupling, dtype=np.int32)
        self.radius = np.asarray(radius_, dtype=np.float32)
        self.value = np.asarray(value_, dtype=np.float32)
        self.rate = np.asarray(rate_, dtype=np.float32)
        self.order = np.asarray(order_, dtype=np.int32)
        self.hide_until_unlocked = np.asarray(hide_, dtype=bool)
        for name in ("radius", "value", "rate"):
            arr = getattr(self, name)
            if np.any(arr <= 0):
                raise ValueError(f"POI {name} must be > 0, got {arr.tolist()}")
        # A POI of the lowest order is never locked. Flagging one alongside
        # later ones is harmless (a block-level default does exactly that); a
        # flag that can act on no POI at all is a mistake worth reporting.
        can_hide = self.hide_until_unlocked & (self.order > self.order.min())
        if self.hide_until_unlocked.any() and not can_hide.any():
            warnings.warn(
                "hide_until_unlocked is set on POIs "
                f"{np.flatnonzero(self.hide_until_unlocked).tolist()}, but none of them "
                "has an order above the lowest, so none is ever locked or hidden.",
                stacklevel=2,
            )

    def init(self, key, env):
        return POIState(
            position=jnp.asarray(self.position),
            coupling=jnp.asarray(self.coupling),
            radius=jnp.asarray(self.radius),
            value=jnp.asarray(self.value),
            rate=jnp.asarray(self.rate),
            remaining=jnp.asarray(self.value),
            order=jnp.asarray(self.order),
            hide_until_unlocked=jnp.asarray(self.hide_until_unlocked),
        )

    def validate(self, env):
        width, height = env.world_size
        x, y = self.position[:, 0], self.position[:, 1]
        outside = (x < 0) | (x > width) | (y < 0) | (y > height)
        if np.any(outside):
            raise ValueError(
                f"POIs {np.flatnonzero(outside).tolist()} lie outside the "
                f"{width} x {height} arena"
            )
        unreachable = self.coupling > env.n_agents
        if np.any(unreachable):
            warnings.warn(
                f"POIs {np.flatnonzero(unreachable).tolist()} need more rovers "
                f"({self.coupling[unreachable].tolist()}) than the env has "
                f"({env.n_agents}); they can never pay.",
                stacklevel=2,
            )
            # A POI that never pays is never used up, so every POI ordered after
            # it stays locked for the whole episode.
            blocked = ~unreachable & (self.order > self.order[unreachable].min())
            if np.any(blocked):
                warnings.warn(
                    f"POIs {np.flatnonzero(blocked).tolist()} are ordered after a POI "
                    "that can never pay, so they never unlock either.",
                    stacklevel=2,
                )
