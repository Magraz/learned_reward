"""POI models. ``StaticPOIs`` is the fixed, config-defined layout.

Every POI carries a consumption ``order`` (its temporal coupling, next to the
spatial ``coupling``). Its order is reached once every POI of a lower order is
used up (``order_reached``). By default (``enforce_order``) a POI can pay only
from then on; ``unlocked`` is the one definition of when a POI may pay, and the
reward and the scripted policies both read it. With ``enforce_order`` false the
order locks nothing and acts only on what the rovers sense. Every POI, used up
or not, is sensed by the rovers, except a POI flagged ``hide_until_unlocked``
until its order is reached and a POI flagged ``hide_when_used_up`` once it is
used up; ``visible`` is the one definition of what the rovers can sense.
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
    "enforce_order",
    "hide_until_unlocked",
    "hide_when_used_up",
}


def order_reached(pois):
    """(..., P) bool: POIs whose temporal coupling is met, i.e. every POI of a
    lower ``order`` is used up. Says nothing about a POI's own budget, nor
    about ``enforce_order``.

    The lowest order among the POIs with budget left is always reached.
    """
    pending = pois.remaining > 0  # (..., P)
    earlier = pois.order[..., None, :] < pois.order[..., :, None]  # [p, q]: q before p
    return ~jnp.any(earlier & pending[..., None, :], axis=-1)


def unlocked(pois):
    """(..., P) bool: POIs their order lets pay now: the order is reached, or
    the POI does not enforce it (``enforce_order`` false). Says nothing about a
    POI's own budget.

    The lowest order among the POIs with budget left is always unlocked, so a
    POI with budget left is always available somewhere.
    """
    return order_reached(pois) | ~pois.enforce_order


def visible(pois):
    """(..., P) bool: POIs the rovers can sense. Every POI is, used up or not,
    except a POI flagged ``hide_until_unlocked`` until its order is reached
    (``order_reached``, whether or not the order is enforced) and a POI flagged
    ``hide_when_used_up`` once it is used up.

    With the order enforced a POI pays only once its order is reached, so
    ``hide_until_unlocked`` hides it only before it pays. Without, rovers that
    meet a hidden POI's coupling are paid (it may even be used up) while it
    stays hidden until its order is reached.
    """
    used_up = pois.remaining <= 0
    return (~used_up | ~pois.hide_when_used_up) & (
        ~pois.hide_until_unlocked | order_reached(pois)
    )


def _flag(entry, name, default, k):
    """An entry's true/false key, falling back to the block default."""
    value = entry.get(name, default)
    if not isinstance(value, (bool, np.bool_)):
        raise ValueError(f"POI {k}: {name} must be true or false, got {value!r}")
    return bool(value)


class StaticPOIs(POIModel):
    """A fixed POI layout, identical in every episode.

    Args:
        entries: one mapping per POI with ``position: [x, y]`` and ``coupling``
            (required) and optional ``radius`` / ``value`` / ``rate``, which
            override the defaults below for that POI, and ``order`` (an
            integer >= 0, default 0): its order is reached once every POI of a
            lower order is used up (see ``order_reached``). With every order
            equal, as by default, POIs can be used in any order. An entry may
            also set ``enforce_order``, ``hide_until_unlocked`` and
            ``hide_when_used_up`` (overriding the defaults below).
        radius: observation radius (inclusive) a rover must be within.
        value: total reward the POI can pay in one episode.
        rate: reward paid per step while its coupling is met; the POI is used
            up after ``ceil(value / rate)`` coupled steps.
        enforce_order: true or false. True (the default): the POI pays only
            once its order is reached (see ``unlocked``). False: its order
            locks nothing; it pays whenever its coupling is met and acts only
            through ``hide_until_unlocked``, which still follows the order.
            The scored team reward is then the unordered layout's.
        hide_until_unlocked: true or false. A flagged POI is not sensed by the
            rovers until its order is reached, i.e. while a POI of a lower
            ``order`` still has budget (see ``visible``), whether or not the
            order is enforced. A POI of the lowest order is never hidden, so
            the flag only acts on a POI with a higher order.
        hide_when_used_up: true or false. A flagged POI is not sensed once it
            is used up. Off by default: a used-up POI is sensed exactly like a
            live one. True is the rule every run before 2026-11-04 trained
            under.
    """

    def __init__(
        self,
        entries,
        radius=1.0,
        value=1.0,
        rate=0.05,
        enforce_order=True,
        hide_until_unlocked=False,
        hide_when_used_up=False,
    ):
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
            rows.append(
                (
                    entry["position"],
                    int(coupling),
                    float(entry.get("radius", radius)),
                    float(entry.get("value", value)),
                    float(entry.get("rate", rate)),
                    int(order),
                    _flag(entry, "enforce_order", enforce_order, k),
                    _flag(entry, "hide_until_unlocked", hide_until_unlocked, k),
                    _flag(entry, "hide_when_used_up", hide_when_used_up, k),
                )
            )
        (position, coupling, radius_, value_, rate_, order_, enforce_, hide_,
         hide_used_) = zip(*rows)
        self.position = np.asarray(position, dtype=np.float32)
        if self.position.shape != (len(rows), 2):
            raise ValueError(f"every POI position must be [x, y], got {position}")
        self.coupling = np.asarray(coupling, dtype=np.int32)
        self.radius = np.asarray(radius_, dtype=np.float32)
        self.value = np.asarray(value_, dtype=np.float32)
        self.rate = np.asarray(rate_, dtype=np.float32)
        self.order = np.asarray(order_, dtype=np.int32)
        self.enforce_order = np.asarray(enforce_, dtype=bool)
        self.hide_until_unlocked = np.asarray(hide_, dtype=bool)
        self.hide_when_used_up = np.asarray(hide_used_, dtype=bool)
        for name in ("radius", "value", "rate"):
            arr = getattr(self, name)
            if np.any(arr <= 0):
                raise ValueError(f"POI {name} must be > 0, got {arr.tolist()}")
        # A POI of the lowest order always has its order reached, so neither
        # order flag acts on it. Setting one alongside later POIs is harmless
        # (a block-level default does exactly that); a flag that can act on no
        # POI at all, or an order that acts on nothing, is a mistake worth
        # reporting.
        later = self.order > self.order.min()
        can_hide = self.hide_until_unlocked & later
        if self.hide_until_unlocked.any() and not can_hide.any():
            warnings.warn(
                "hide_until_unlocked is set on POIs "
                f"{np.flatnonzero(self.hide_until_unlocked).tolist()}, but none of them "
                "has an order above the lowest, so none is ever locked or hidden.",
                stacklevel=2,
            )
        unenforced = ~self.enforce_order
        if unenforced.any() and not (unenforced & later).any():
            warnings.warn(
                "enforce_order is false on POIs "
                f"{np.flatnonzero(unenforced).tolist()}, but none of them has an "
                "order above the lowest, so the flag changes nothing.",
                stacklevel=2,
            )
        if later.any() and not (later & (self.enforce_order | self.hide_until_unlocked)).any():
            warnings.warn(
                f"POIs {np.flatnonzero(later).tolist()} have an order above the "
                "lowest, but none enforces it or hides until it is reached, so "
                "the order has no effect.",
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
            enforce_order=jnp.asarray(self.enforce_order),
            hide_until_unlocked=jnp.asarray(self.hide_until_unlocked),
            hide_when_used_up=jnp.asarray(self.hide_when_used_up),
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
            # A POI that never pays is never used up, so no POI ordered after it
            # ever has its order reached: it stays locked (enforced) or hidden
            # (hide_until_unlocked) for the whole episode.
            blocked = ~unreachable & (self.order > self.order[unreachable].min())
            never_unlock = blocked & self.enforce_order
            if np.any(never_unlock):
                warnings.warn(
                    f"POIs {np.flatnonzero(never_unlock).tolist()} are ordered after a "
                    "POI that can never pay, so they never unlock either.",
                    stacklevel=2,
                )
            never_seen = blocked & ~self.enforce_order & self.hide_until_unlocked
            if np.any(never_seen):
                warnings.warn(
                    f"POIs {np.flatnonzero(never_seen).tolist()} are ordered after a "
                    "POI that can never pay, so they are never sensed.",
                    stacklevel=2,
                )
