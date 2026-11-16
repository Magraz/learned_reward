# Experiment 1: Jointly learning a shared POI visibility mask

**Status:** Proposed experiment; no training runs or results are reported here.  
**Date:** November 13, 2026

## Research question

Can a shared visibility policy, learned alongside rover policies, improve cooperative task performance over fixed visibility rules while preserving the original task reward?

The controller chooses which points of interest (POIs) every rover can observe. Rovers learn how to act on those observations. Both policies are trained from scratch, and the controller remains active during evaluation.

The first experiment isolates selection among productive POIs. It does not attempt to establish optimality, generalization to new layouts, or performance without the controller.

## Hypothesis and primary comparison

**Hypothesis:** Selectively revealing productive POIs can help the rovers gather at a common target and transition between targets, improving learning speed or final performance.

The primary comparison is learned masking versus revealing all remaining productive POIs. Both conditions automatically hide depleted POIs, so their difference cannot be explained solely by communicating depletion through disappearance.

A secondary comparison asks whether learned masking can match or exceed a hand-designed visitation sequence.

## Task configuration

Use a controlled variant of `conf/env/rover_3a_3poi_c3_implicit.yaml`.

| Setting | Proposed value |
|---|---|
| Arena | 10 × 10 |
| Rovers | 3 |
| POIs | 3 |
| POI positions | P1: (5, 4); P2: (2, 6); P3: (9, 9) |
| Coupling | All 3 rovers required at each POI |
| Collection radius | 1.0 for every POI |
| Total value | 1.0 per POI |
| Payout rate | 0.05 per coupled step per POI |
| Episode limit | 300 environment steps |
| Starting positions | Existing random spawn region: x ∈ [2, 8], y ∈ [0, 2] |
| Dynamics | Existing rover dynamics; maximum speed 0.5 and maximum turn π/4 |
| Observations | Existing heading-relative sectors: 8 rover sectors and 8 POI sectors |
| Sensor radius | Existing arena-diagonal setting |
| Reward | Original shared team payout |
| Reward ordering | Disabled in every condition |
| Termination | All POIs depleted, or episode limit reached |

Equal POI values, payout rates, and radii make gathering and transitioning the initial coordination problem. Heterogeneous values and coupling requirements are deferred to subsequent experiments.

### Horizon check

The existing three-POI configuration inherits a 100-step episode limit, while its POIs require 100, 150, and 60 coupled steps to deplete. With all three rovers required at each spatially separated POI, collecting everything is infeasible within that horizon.

The proposed configuration requires 20 coupled steps per POI, or 60 collection steps plus navigation. Before training, verify with a competent reference trajectory that the 300-step horizon accommodates full completion over representative starts. This validates feasibility; it does not establish an optimal route. If necessary, revise the horizon once and apply the same value to every condition before comparative training.

## Shared mask definition

Let `m_t[p] = 1` mean POI `p` is visible at step `t`. The controller selects a nonempty subset of the productive POIs, where productive means remaining budget is positive.

When all three POIs are productive, the available masks are:

`100`, `010`, `001`, `110`, `101`, `011`, `111`.

Use a categorical policy over these seven patterns. Exclude any pattern that includes a depleted POI or reveals no productive POI. With two productive POIs there are three admissible patterns; with one there is one. When no productive POIs remain, the episode terminates.

This design deliberately gives the main conditions an automatic depletion filter. The learned component decides which productive POIs to reveal, rather than learning whether depleted POIs should disappear.

### Observation semantics

- Sample one shared mask per environment step, before constructing rover observations.
- Apply it to individual POIs before sector aggregation.
- Keep the existing normalization by the original POI count, so hiding a POI does not rescale the others' contributions.
- Every rover receives the same POI selection, expressed through its own position and heading.
- Do not append the mask or controller state to rover observations.
- Leave rover-to-rover observations unchanged.
- Hidden POIs remain physically present and pay normally when their coupling requirement is met.
- Allow the all-remaining-visible mask; do not require information removal.

## Controller and rover policies

### Mask controller

Use a small feedforward MLP with categorical output over mask patterns. Its inputs are current rover positions and headings, POI positions and remaining budgets, and time remaining in the episode. Use fixed POI indexing for this initial fixed-layout experiment.

The controller has global task information at execution time. Accordingly, the system should be described as having centralized observation control. The experiment does not establish that masking is better than every other way of providing this information to the rovers.

### Rover policies

Retain the existing recurrent MAPPO configuration with parameter sharing among rovers. The actors receive only their local masked observations and maintain their usual recurrent states. Keep architecture and critic information consistent across experimental conditions.

## Joint training protocol

Train a separate controller actor with PPO and its own value baseline. Train the rover actor and critic using the existing recurrent MAPPO setup. Both policies learn from the same on-policy rollouts.

For each environment step:

1. Observe the current task state with the controller.
2. Sample an admissible shared mask.
3. Construct masked rover observations.
4. Sample rover actions and advance the environment.
5. Record the original team reward and the information required to update both policies.

After collecting a rollout, update the controller and rover policies from that collected experience. The controller learns through the return associated with its sampled mask; no gradient through the discrete mask is required. Preserve the sampled masks and resulting rover observations when computing policy updates.

Both policies optimize the original team reward with the same discount factor. Initially use no auxiliary reward for hiding, no visibility budget penalty, and no mask-switching penalty. Standard policy entropy regularization may be used, with the controller coefficient specified and held fixed across runs.

Retain the current rover training defaults for the initial experiment, including discount factor 0.99. This favors earlier reward, so report undiscounted episode reward separately and avoid describing the training objective as pure undiscounted reward maximization.

## Experimental conditions

Train fresh rover policies separately under every visibility rule. Do not evaluate all conditions using rovers trained only with learned masks.

| Condition | Visibility rule | Purpose |
|---|---|---|
| Full visibility | All POIs visible, including depleted POIs | Connect to the existing observation baseline |
| Remaining POIs | All productive POIs visible; depleted POIs hidden | Primary baseline controlling for depletion information |
| Fixed sequence | Only the current target visible; reveal the next after depletion | Hand-designed guidance baseline |
| Random masks | Uniformly sample admissible nonempty masks each step | Control for arbitrary filtering |
| Learned masks | Sample the jointly trained controller's admissible mask | Proposed method |

For the fixed sequence, use P3 → P2 → P1, matching the order in the existing three-POI hiding configuration. Fix this choice before inspecting learned-mask results. If a later POI is depleted while hidden, skip it when advancing the sequence. Reward availability remains unordered.

The random baseline shares the learned method's admissible action set and decision frequency. It does not necessarily match the learned policy's visibility-count or switching distribution. If that distinction becomes important, add a matched random control as a follow-up.

## Training and evaluation budget

| Item | Initial protocol |
|---|---|
| Conditions | 5 |
| Independent training seeds | 5 per condition |
| Interaction budget | 10 million environment steps per run |
| Total planned training interactions | 250 million environment steps |
| Parallel environments | Retain 32 per run |
| Evaluation starts | Identical held-out reset seeds across conditions |
| Final evaluation | 100 episodes per trained model |
| Checkpoint for primary final comparison | Final checkpoint at the fixed interaction budget |

Count an environment step as one joint rover transition, not one transition per rover. Keep evaluation interactions separate from the training budget. Report wall-clock cost in addition to interactions because learned masking adds computation.

Use a fixed evaluation schedule across conditions and preserve the same rover evaluation convention. For the primary learned-mask evaluation, choose the highest-probability admissible mask; evaluate random masking with its prescribed random sampling. A secondary evaluation with sampled learned masks can reveal dependence on stochastic execution.

Aggregate results across independent training seeds. Evaluation episodes estimate each trained model's performance; they are not independent training replications. Report seed-level values alongside means and uncertainty intervals, especially given the small number of seeds.

Do not choose a reported checkpoint using the final held-out evaluation episodes. Any tuning should use separate development evaluations with comparable tuning effort across conditions.

## Measurements

### Performance

- **Primary endpoint:** Final mean undiscounted team reward, bounded by 3 in this configuration.
- **Learning efficiency:** Reward versus training interactions and area under that learning curve over the fixed budget.
- **Completion rate:** Fraction of episodes depleting all three POIs.
- **Completion time:** Steps to full depletion on successful episodes, always reported alongside completion rate. Also report an episode-length-capped time across all episodes to expose failures.
- **First coordination time:** Steps until any POI first meets its coupling requirement; report episodes that never reach this event.

### Controller behavior

- Number of visible productive POIs over time.
- Mask-switching frequency, separating switches caused by admissibility changes at depletion from discretionary switches.
- Time spent revealing a single target versus multiple targets.
- Mask changes around POI collection and depletion.
- Collection order, rover trajectories, and payouts from hidden POIs.

Create an illustrative figure aligning rover trajectories, POI payouts, and a three-row visibility timeline. Use a stated episode-selection rule, such as fixed evaluation seeds, rather than showing only successful examples.

## Interpretation criteria

| Outcome | Supported interpretation |
|---|---|
| Learned masks improve over all remaining POIs | Evidence that jointly learned selective visibility adds value beyond the shared depletion filter |
| Learned masks only improve over full visibility | Improvement could be explained by depletion information alone |
| Learned masks match the fixed sequence | Useful guidance can be recovered through joint learning on this task |
| Learned masks exceed the fixed sequence | Learned guidance improves over that particular prescribed order, without establishing optimality |
| Learned masks match random masks | No demonstrated advantage from learning the filtering rule under this protocol |
| Controller usually reveals all remaining POIs | The task or learned solution provides little evidence that selective hiding is useful |
| Final reward saturates across conditions | Learning efficiency and completion time may distinguish methods; final reward alone does not |
| Controller changes masks rapidly | Inspect whether masks act as a communication signal rather than sustained target selection |

The strongest initial result would be a reproducible improvement over the remaining-POIs baseline, accompanied by interpretable coordination behavior. A single successful seed is insufficient.

## Limits and follow-up experiments

This is a feasibility experiment on one fixed layout. It does not demonstrate transfer, globally optimal behavior, or rover independence from the controller. The controller's global information access is part of the method.

If the initial comparison is promising, prioritize:

1. A rover baseline with explicit POI status, and comparisons that account more fully for the controller's information advantage.
2. Heterogeneous POI values, rates, radii, and coupling requirements.
3. Held-out layouts and broader starting-state distributions.
4. A slower mask decision interval if rapid switching dominates behavior.
5. Learning whether to hide depleted POIs, removing the automatic depletion filter.
6. Agent-specific masks for assigning different rover groups to different POIs.

Controller-removal or mask-replacement evaluations can diagnose dependence, but they create observation distributions the rover policies may not have been trained on. Treat them as diagnostics rather than substitutes for separately trained baselines.

## Research positioning

Related work already studies learning communication from task rewards. For example, [TarMAC: Targeted Multi-Agent Communication](https://proceedings.mlr.press/v97/das19a.html) learns message content and recipients using downstream task reward. The eventual paper should therefore establish what is distinctive and useful about controlling task-object visibility, rather than claiming novelty for jointly learning an information channel in general.

A claim appropriate to a positive first result would be:

> On a cooperative rover task, a jointly trained shared observation controller improves coordination through selective POI visibility while preserving the task reward.
