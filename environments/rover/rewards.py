"""Default rover reward: coupled POIs that pay a steady rate until used up."""

import dataclasses

import jax
import jax.numpy as jnp

from environments.rover.components import RewardModel, RewardOutput

# A budget below this is snapped to exactly 0, so float32 accumulation cannot
# leave a ~1e-8 residue that pays one extra step (value 1.0 / rate 0.05 must
# deplete in exactly 20 coupled steps).
DEPLETION_TOL = 1e-6


def _payout(observed, pois):
    """Which POIs pay, and how much, when ``observed`` rovers are in each radius.

    ``observed`` is ``(..., P)``; leading axes broadcast, so one call also
    scores a batch of counterfactual counts.
    """
    active = (observed >= pois.coupling) & (pois.remaining > 0)
    return active, jnp.where(active, jnp.minimum(pois.rate, pois.remaining), 0.0)


MODES = ("shared_reward", "difference_rewards")


class CouplingDepletionReward(RewardModel):
    """Team reward from POIs whose coupling is met.

    Args:
        mode: the per-rover reward ``step`` returns, i.e. what every rover is
            trained on, named by its ``info`` key: ``"shared_reward"`` (G for
            every rover) or ``"difference_rewards"`` (D_i). The team reward G
            is ``info["task_reward"]`` under both, and both signals are in
            ``info`` under both.

    Per step, on the rovers' post-move positions:

    * a rover observes POI p when ``||rover - p|| <= radius_p`` (inclusive);
    * p is active when at least ``coupling_p`` rovers observe it and it has
      budget left. Rovers beyond the coupling add nothing;
    * an active POI pays ``min(rate_p, remaining_p)`` and its budget drops by
      that much, so it pays a steady rate until used up and then nothing for
      the rest of the episode;
    * the team reward is G = sum of payouts; ``info["shared_reward"]`` is G
      for every rover;
    * ``info["difference_rewards"]`` is each rover's difference reward
      D_i = G - G_{-i}, where G_{-i} is this step's team reward with rover i
      removed and everything else (the other rovers, the POI budgets) kept.
      Removing rover i lowers the count of every POI it observes by one, so
      D_i is the payout of the POIs whose coupling rover i completes exactly.
      A rover beyond a POI's coupling earns 0 from it, and so does one waiting
      at a POI whose coupling is not yet met. The D_i need not sum to G;
    * the episode terminates once every POI is used up.
    """

    def __init__(self, mode="shared_reward"):
        if mode not in MODES:
            raise ValueError(f"unknown reward mode {mode!r}; known: {list(MODES)}")
        self.mode = mode
        self.per_agent_rewards = mode != "shared_reward"

    def __call__(self, env, rovers, pois):
        dist = jnp.linalg.norm(
            rovers.position[:, None, :] - pois.position[None, :, :], axis=-1
        )  # (A, P)
        in_radius = dist <= pois.radius[None, :]
        observed = in_radius.sum(axis=0)  # (P,)
        active, payout = _payout(observed, pois)
        remaining = pois.remaining - payout
        remaining = jnp.where(remaining < DEPLETION_TOL, 0.0, remaining)

        shared_reward = jnp.full((dist.shape[0]), payout.sum())

        # Row i of the counts: every POI's count with rover i removed. The
        # difference is taken per POI before summing, so a POI rover i does not
        # complete contributes exactly 0 (no float residue from G - G_{-i}).
        _, payout_without = _payout(
            observed - in_radius.astype(observed.dtype), pois
        )  # (A, P)
        difference_rewards = (payout - payout_without).sum(axis=-1)  # (A,)
        per_rover = {
            "shared_reward": shared_reward,
            "difference_rewards": difference_rewards,
        }
        return RewardOutput(
            reward=per_rover[self.mode],
            team_reward=payout.sum(),
            pois=dataclasses.replace(pois, remaining=remaining),
            terminated=jnp.all(remaining <= 0.0),
            info={
                **per_rover,
                "poi_observed": observed,
                "poi_active": active,
                "poi_payout": payout,
            },
        )
