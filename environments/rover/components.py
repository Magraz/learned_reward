"""Interfaces of the pluggable parts of ``BaseRoverEnv``.

A rover-domain variant is a choice of three components:

* ``POIModel`` — where the points of interest are and what they are worth
  (``init``), and how they evolve between steps (``step``, identity by default).
* ``ObservationModel`` — what each rover sees.
* ``RewardModel`` — what the team is paid and how POIs are used up. The reward
  model owns POI consumption because depletion is part of the reward's
  semantics, so it returns the updated ``POIState`` and the termination flag.
  It returns two rewards: one per rover, the signal the learner trains on, and
  the team's, which evaluation scores. They stay separate because a per-rover
  signal need not sum to the team's (difference rewards do not).

Every method is called inside ``jit``/``vmap``, so it must be pure jnp on the
state it is given. Static configuration (couplings from a yaml, sector counts)
lives on the component object.
"""

import abc
import dataclasses

import jax

from environments.rover.state import POIState, RoverEnvState, RoverState


class POIModel(abc.ABC):
    @abc.abstractmethod
    def init(self, key, env) -> POIState:
        """POIs at the start of an episode (key: per-episode randomness)."""

    def step(self, pois: POIState, rovers: RoverState, t) -> POIState:
        """POIs after the rovers moved, before the reward. Identity by default;
        override for moving or appearing POIs."""
        return pois

    def validate(self, env) -> None:
        """Python-level checks against the env, run once at construction."""


class ObservationModel(abc.ABC):
    obs_dim: int  # per-agent observation width

    @abc.abstractmethod
    def __call__(self, env, state: RoverEnvState) -> jax.Array:
        """(n_agents, obs_dim) observations of ``state``."""


@dataclasses.dataclass(frozen=True)
class RewardOutput:
    reward: jax.Array  # (n_agents,) the reward each rover is trained on
    team_reward: jax.Array  # () what the team earned; evaluation scores this
    pois: POIState  # POIs after consumption
    terminated: jax.Array  # () bool, nothing left to earn
    info: dict  # extra per-step quantities, merged into ``step``'s info


class RewardModel(abc.ABC):
    # False when every rover always receives the same reward, so a trainer may
    # keep a single team value instead of one per rover.
    per_agent_rewards: bool = True

    @abc.abstractmethod
    def __call__(self, env, rovers: RoverState, pois: POIState) -> RewardOutput:
        """Reward for the rovers' post-move positions."""
