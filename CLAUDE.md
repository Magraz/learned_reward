# When user gives instructions, push back if you think the user is wrong. Do not accept everything the user says as source truth. Use your best judgement but share your reasoning with the user and provide both options. Always go with what the user chooses after this.

Whenever building new code, try to reuse as much code as possible. If the new functionality overlaps heavily with other parts of the code, find a way to abstract and reuse the logic instead of duplicating the functionality.

Always keep the CLAUDE.md file up to date to reflect the current functionality and architecture of the code.

Avoid abbreviating terms, and if you are going to use an abbreviation, explain it don't assume I know what an abbreviated term is.

When analyzing results, explain to me the quantities that are being logged, their meaning as well as why use them.

## Repository state (re-initialized 2026-10-30)

This repository was re-initialized on 2026-10-30 as one commit
(`5dec808 first commit`). It holds a pruned copy of the earlier
`morphology-learning` project. `pyproject.toml`, the committed Hydra artifact
`multirun.yaml` and the per-trial cluster scripts still name that project.

What remains:
- **Training stacks:** `mappo_jax` and `rmappo_jax` (JAX). MAPPO =
  Multi-Agent Proximal Policy Optimization: PPO with a centralized critic.
- **Environments:** only the rover domain (`environments/rover/`), plus the
  shared sector binning in `environments/sectors.py`.
- **Config (`conf/`):** algorithm groups `mappo_jax` and `rmappo_jax`, 5 rover
  env groups, 3 model groups (`mlp` for `mappo_jax`; `rmappo`, `mappo_mlp` for
  `rmappo_jax`), the seed list and the joblib launcher.
- **Tooling:** `train.sh`, `train_batches.sh`, `scripts/` (cluster launchers,
  progress helper, notebook cleaner), `plotting/`, and the project skill
  `.claude/skills/explain-diff`.

Removed, code and documentation alike: the PyTorch stacks `mappo` and
`mappo_vanilla` (deleted 2026-10-30, after the re-initialization), the Box2D
suite, the MJX (MuJoCo XLA) suite, the SMAX / SMAClite / SMACv2 StarCraft
wrappers, `feudal_mappo_jax`, `simplified_feudal_mappo_jax`, the hierarchical
skill controller, DCG (Deep Coordination Graphs) and its macro-action variant,
the oracle difference rewards, and the `hypergraphs/` package. Their
documentation, every measured result included, is the CLAUDE.md of commit
`5dec808` (`git show 5dec808:CLAUDE.md`); the torch stacks' code is in that
commit too. Results of removed environments are still on disk under
`experiments/results/` (git-ignored).

### Fixes applied 2026-10-30

The pruning had left the JAX stacks unable to run. Fixed:
- `conf/config.yaml` defaults are now `algorithm: rmappo_jax`,
  `env: rover_2poi_c2`, `model: rmappo` (they named deleted groups, so every
  composition without `model=` failed, including 20 of the 35 rover tests).
  `uv run python train.py` with no arguments now trains.
- `mappo_jax/run.py:make_env` builds only the rover domain. The MJX and SMAX
  branches imported deleted modules at module level and compared against
  deleted `EnvironmentEnum` members. `view()` lost its MJX and SMAX rendering
  paths and raises `NotImplementedError` for a non-rover env.
- `mappo_jax/trainer.py` lost the decision-aligned difference-reward path,
  which imported the deleted macro wrapper.
- `TrainingStatsTracker` moved, unchanged, from the deleted
  `mappo_vanilla/trainer_components/stats_tracker.py` to
  `algorithms/mappo_jax/stats_tracker.py`.
- `conf/model/mlp.yaml` (`hidden_dim: 168`) restored from the old project as
  `mappo_jax`'s model group.
- `test_every_rover_variant_composes_and_builds` now allows a variant its own
  `sensor_radius`.

Verified: all 66 tests in `algorithms/tests` pass. On the CPU,
`rmappo_jax` (defaults) trains, resumes with `checkpoint=true`, evaluates and
renders; `mappo_jax` with `model=mlp` trains and renders.

### Loose ends (not fixed)

- **Env groups renamed without their callers (commit `2fa6f43`, 2026-11-02).**
  `rover_2poi_c2` → `rover_4a_2poi_c2`, `rover_2poi_c8` → `rover_16a_2poi_c8`,
  `rover_4poi_c1234` → `rover_4a_4poi_c1234`, `rover_6poi_c3_ring` →
  `rover_6a_6poi_c3`, `rover_6poi_c8-4-4-8-4-4` → `rover_16a_6poi_c844844`; the
  `_diffrew` groups were deleted. Still on the old names: `conf/config.yaml`'s
  default env (so a bare `uv run python train.py` fails), the tests (30 fail
  with `Could not find 'env/rover_2poi_c2'` and the like, found 2026-11-03),
  `ROUTE_PLANS` and `scripted_steps.py`, docstrings, the launch lists in
  `train_batches.sh` and `scripts/hpc/experiment_batch*.sh`, and the variant
  names in this file. The base group's `rate` is now 0.1 (was 0.05), so tests
  timing a variant's depletion need new numbers once they run again.
  Renamed again on 2026-11-03 (uncommitted): `_ord<levels>` → `_ord` (e.g.
  `rover_4a_2poi_c2_ord01` → `rover_4a_2poi_c2_ord`) and `_implicit012` →
  `_implicit`; `ROUTE_PLANS` still names `rover_4a_3poi_c3_implicit012`.
- **`_ord` comments copied from the unordered groups** (found 2026-11-03):
  `rover_4a_4poi_c1234_ord` says the POIs "can be worked in parallel by
  sub-teams" and `rover_6a_6poi_c3_ord` that "two POIs can be worked at
  once", but their strict orders (0-1-2-3, 0-…-5) let only one POI pay at a
  time. `rover_4a_4poi_c1234_ord` also senses globally where
  `rover_4a_4poi_c1234` has `sensor_radius: 2.0`, so that pair differs in
  sensing as well as in order.

- `conf/model/mlp.yaml` is missing again (found 2026-10-30, after the restore
  above; never committed), so `algorithm=mappo_jax model=mlp` fails with
  `Could not find 'model/mlp'`. Its content was `# @package _global_` and
  `model_params: {hidden_dim: 168}`.

- `conf/algorithm/mappo.yaml` and `mappo_vanilla.yaml` remain, but `_dispatch`
  has no case for them: `algorithm=mappo` ends in an `UnboundLocalError` on
  `runner`.
- `algorithms/runner.py`, `algorithms/utils.py`, `algorithms/create_env.py` and
  `algorithms/tests/_rollout_utils.py` were used only by the torch stacks and
  are imported by nothing (`_rollout_utils.py` imports the deleted `mappo`).
- `mappo_jax/mappo.py`'s `squash`, `advantage_correction` and
  `masked_statistics` arguments, and `MAPPOCritic.head_init_scale`: their only
  callers were the feudal stacks. All default to off. The env hooks
  (`avail_actions`, `global_state`, `info["active"]`, per-agent rewards) are
  generic and kept, though no current env uses them.
- `environments/types.py:EnvironmentParams` is never instantiated.
- `requirements.txt` starts with `-e ./smaclite`, which does not exist, so the
  README's `uv pip install -r requirements.txt` fails. `uv sync` (from
  `pyproject.toml`) works.
- `pyproject.toml` depends on `mujoco-mjx`, `mvlearn`, `seaborn`, `dill`,
  `gymnasium[box2d]`, `pygame-ce` and `scikit-learn`, which no live code
  imports. torch is still needed: `test_rmappo_jax.py` checks the GRU against
  `torch.nn.GRU`. `jaxmarl` is used only by the standalone MPE demo script.
- The virtual environment has both `jax-cuda12-*` and `jax-cuda13-*` plugins.
  Every run prints a `PJRT_Api already exists for device type cuda` traceback
  while JAX discovers them; it is non-fatal.
- The README's "Feudal goal visualization" section documents removed code.
- `conf/algorithm/rmappo_jax.yaml` says `n_total_steps` is "the repo's 1e8";
  the value is 1e7.
- Launch scripts: `scripts/hpc/experiment_batch.sh`, `experiment_set.sh`,
  `experiment.sh` and `experiment_trial.sh` list removed groups;
  `run_trial_cpu.sh` and `run_trial_cpu_evaluate.sh` call the retired
  `run_trial.py`; `run_trial_gpu.sh`, `run_trial_gpu_wildwood.sh` and
  `run_watchdog*.sh` hardcode `.../morphology-learning/...` paths on the
  cluster. Set up for the rover groups: `train_batches.sh`,
  `experiment_batch_node.sh`, and `run_node_gpu.sh`, which finds the repository
  relative to itself.

## Experiment config: Hydra is the sole path (`conf/` + `train.py`)

Runs are launched only through the Hydra entry point `train.py`. It composes a
run from orthogonal groups (**algorithm × env × model × seeds**), resolves it,
and hands it to `_dispatch(algorithm, exp_dict, env_config, batch_dir,
results_dir, trial_id, view, checkpoint, evaluate, debug)` in
`algorithms/algorithms.py`. `_dispatch` builds the algorithm's
`Experiment(**exp_dict)`, constructs its Runner, and calls `train()`, `view()`
or `evaluate()`.

- Dispatch cases: `mappo_jax` (`MAPPO_JAX_Runner`) and `rmappo_jax`
  (`RMAPPO_JAX_Runner`). `debug=true` disables JAX compilation
  (`jax_disable_jit`).
- Paths: results go to
  `experiments/results/<env group>/<model group>/<trial_id>/{logs,models}`. The
  algorithm is not in the path, so two algorithms must never share a model
  group name. `batch_dir` (`experiments/yamls/<env group>`) is a legacy
  argument nothing reads now.
- A numeric `trial_id` indexes `params.random_seeds` (`conf/seeds/standard.yaml`,
  30 seeds). Any other id uses the first seed (118). Use a non-numeric id
  (e.g. `trial_id=smoke`) for smoke runs.
- `conf/config.yaml`: the defaults list (`algorithm: rmappo_jax`,
  `env: rover_2poi_c2`, `model: rmappo`, `seeds: standard`, `_self_`) and the
  top-level `device` (`cpu`), `trial_id`, `view`, `checkpoint`, `evaluate`,
  `debug` and `n_jobs`. The defaults must stay a runnable combination: the
  rover tests compose this file too. Group order sets precedence (later wins):
  the algorithm supplies the base `params` / `model_params`, the env group may
  override `params`, the model group overrides `model_params`, and the seeds
  group injects `params.random_seeds`. `hydra.job.chdir=false`,
  `output_subdir: null` and empty log handlers keep the working directory and
  results paths untouched.
- Every group file must start with `# @package _global_` and put
  `params:` / `model_params:` / `env:` at column 0. Without the header Hydra
  nests the keys under the group's own node, where nothing reads them, and the
  run silently uses the algorithm defaults. This cost 12 full runs once.
- Groups present:
  - `conf/algorithm/{mappo_jax,rmappo_jax}.yaml` (plus the stale
    `mappo.yaml` / `mappo_vanilla.yaml`, see Loose ends);
  - `conf/env/rover_*.yaml` (5 variants, see the rover section);
  - `conf/model/mlp.yaml` (`mappo_jax`), `conf/model/{rmappo,mappo_mlp}.yaml`
    (`rmappo_jax`). Each stack rejects the other's groups with a `TypeError`
    before any results directory is created;
  - `conf/seeds/standard.yaml`;
  - `conf/hydra/launcher/joblib_auto.yaml`.
- Run, and sweep (add `hydra/launcher=joblib_auto` for local parallelism):
  ```
  uv run python train.py                                   # rmappo_jax, rover_2poi_c2, rmappo
  uv run python train.py algorithm=mappo_jax model=mlp trial_id=0
  uv run python train.py -m env=rover_4poi_c1234 model=rmappo,mappo_mlp trial_id=0,1,2
  ```
  A `-m` sweep writes `multirun.yaml` into the repository root, because the
  sweep directory is `.`. The committed `multirun.yaml` is such an artifact, not
  config.
- **Adding an env group:** create `conf/env/<name>.yaml` with an `env:` block
  naming an `environment` that `mappo_jax.run.make_env` builds (add a branch and
  an `EnvironmentEnum` member for a new environment). The rover builder
  validates its keys (`make_rover_env` raises on any unknown key). A builder
  that pulls named keys with `.get()` ignores a misspelled or misindented key
  silently; that once doubled a group's batch through an `n_steps` indented
  under `env:`.
- **Hardware must not change the optimization trajectory.** The machine decides
  how fast data is gathered, never what the optimizer sees.
  1. **The batch is config, not hardware.** The JAX stacks' per-update batch is
     `params.n_steps * env.n_envs` (`n_steps` is the static length of the
     jitted collection scan), so `n_envs` must be a literal in the env group.
     (The removed torch stacks had an explicit `params.batch_size` instead,
     because their `n_envs` was a core count.)
  2. **`n_envs` lives in the env group, never in `_self_`.** Being last in the
     defaults list, a value in `conf/config.yaml` would override every group.
     The rover groups set a literal `n_envs: 32`, a vmap width and therefore a
     hyperparameter. Keep it identical across compared arms.
- **Parallelism.** `train.py` registers two resolvers at import: `${cores:}`
  (usable cores from the CPU-affinity mask, so it respects `taskset`, cgroup
  and SLURM limits) and `${envs_per_job:<n_jobs>}` (`max(1, cores // n_jobs)`).
  No current group uses them; they served subprocess environments.
  `hydra/launcher=joblib_auto` runs `n_jobs` sweep jobs at once.
  `MAPPO_JAX_Runner` deliberately does not import torch.

### Cluster launch (`scripts/hpc/`, see `usage.md`)

See Loose ends for which scripts still work.
- `run_trial_gpu.sh` / `run_trial_gpu_wildwood.sh` submit
  `train.py ... checkpoint=true` under the job name
  `<trial>_<model>_<algorithm>_<batch>`. Both source `gpu_job_preamble.sh`
  (job info, `module load`, `nvidia-smi` check).
- `experiment_batch.sh` writes a ledger,
  `job_tracking/<algorithm>_<batch>_latest.csv`, once at launch (rows through
  `submit_and_record.sh`), and starts `watchdog_loop.py`, which runs
  `watchdog_jobs_v2.sh` every 10 minutes.
- **Watchdog.** A ledger job counts as done when `sacct` reports `COMPLETED`
  for its ledger ID **or** for any job with its name submitted since the
  ledger's timestamp. Before 2026-10-15 only the ledger ID was checked, so a job
  resubmitted after `TIMEOUT` was resubmitted every cycle forever. Each such
  rerun loaded the finished checkpoint, ran no updates and exited 0. A failed
  `sacct` query aborts the cycle instead of resubmitting every job.
  `COMPLETED` only means `train.py` exited 0; any other final state, a config
  error included, is resubmitted on every cycle.
- **All trials in one node job (`experiment_batch_node.sh`).** For trials small
  enough that many share a GPU (rover).
  - Edit `RUN_NAME` and the lists at its top (currently `RUN_NAME=rover`, the 5
    rover groups × `mappo_mlp`/`rmappo` × trials 0-9, `rmappo_jax`), then
    `bash scripts/hpc/experiment_batch_node.sh`.
  - It writes `job_tracking/<algorithm>_node_<RUN_NAME>_trials.tsv`, submits one
    job `0_node_<algorithm>_<RUN_NAME>` through `run_node_gpu.sh`, writes a
    one-row ledger, and starts a watchdog with
    `run_watchdog.sh --run-script run_node_gpu.sh`.
  - It refuses to launch while that job or its watchdog is queued: two jobs on
    one list would train the same trials into one results directory.
  - The job runs `uv sync` and a JAX GPU check once, sets `UV_NO_SYNC=1`, then
    runs `train_batches.sh` with `TRIALS_FILE`, `GPUS` from its own
    `CUDA_VISIBLE_DEVICES` and `N_PARALLEL=TRIALS_PER_GPU`.
  - Resources (top of `run_node_gpu.sh`): 1 GPU on `dgxh-[1-4]` (`preempt`
    partition, `--requeue`), 16 trials per GPU, 20 cores, 96 GB, 24 h,
    progress every 600 s. Sizing, from a 16-rover `rmappo` trial: 2.6 GB of GPU
    memory, 2.4 GB of RAM, about one core and 1.2 s per update. Speed with 16
    trials on one H100 is not measured.
  - Restart: a preemption requeues the job; a timeout or failed trial gets a
    watchdog resubmission (`COMPLETED` needs every trial to finish). Both rerun
    the whole list with `checkpoint=true`, so finished trials exit at once.
  - `run_node_gpu.sh` takes the watchdog's five ledger columns and refuses any
    experiment column other than `node`.
  - Unverified on the cluster: `--nodelist=dgxh-[1-4]` with `--nodes=1`, and
    whether SLURM's kill reaches the trials' process groups under
    `ProctrackType=proctrack/pgid` (`cgroup` is fine).
  - Tested 2026-10-28 with fake `sbatch`/`squeue`/`sacct` (launch, duplicate
    guard, watchdog cycles), and the generated job on the local GPU, including
    SIGTERM, requeue resume and an all-finished rerun.

### Local multi-trial launch (`train_batches.sh`, repository root)

The local counterpart of `experiment_batch.sh`. Edit `BATCH_NAMES` /
`EXPERIMENT_NAMES` / `ALGORITHM` / `TRIAL_START..TRIAL_END` at its top
(currently the 5 rover groups × `mappo_mlp`, `rmappo` × trials 0-9 under
`rmappo_jax`), then `bash train_batches.sh`.
- It calls `train.sh` once per trial. `train.sh` passes `checkpoint=true`, so a
  re-run resumes unfinished trials, and it exits with the trainer's status.
  Both launchers check that every named config file exists first
  (`scripts/check_conf_groups.sh`).
- Environment variables:
  - `N_PARALLEL=<k>`: k trials at once on the GPU. `0` = CPU slots only.
  - `CPU_SLOTS="4 6"`: extra slots on the CPU, one per space-separated entry,
    each a comma-separated list of logical CPUs (`taskset -c` numbering). A CPU
    trial runs under `JAX_PLATFORMS=cpu` and `taskset`. GPU slots fill first.
    The script refuses malformed slots, unusable CPUs and duplicates, and warns
    about efficiency cores (`/sys/devices/cpu_atom/cpus`, Intel hybrid CPUs).
  - `GPUS="0 1"`: `N_PARALLEL` slots per listed GPU, each trial with
    `CUDA_VISIBLE_DEVICES` set to its slot's GPU.
  - `TRIALS_FILE=<tsv>` (with `ALGORITHM` in the environment): replaces the
    lists. Header `batch<TAB>experiment<TAB>trial`, then one trial per line.
  - `DRY_RUN=1`: list the trials only.
  - `PROGRESS_INTERVAL` (default 300 s, `0` = off).
- **It prints nothing once a run starts** (the author's request, 2026-10-28).
  Only errors before any trial runs, and the dry-run listing, reach the
  terminal. Everything goes to `logs/train_batches/<timestamp>/`:
  `<batch>__<experiment>__<trial>.log` per trial, `train_batches.log` (plan,
  warnings, started/finished lines, final summary), `progress.log`,
  `summary.tsv` (status and device per trial: `gpu`, `gpu:<id>` or
  `cpu:<CPUs>`), `trials.tsv`, `train_batches.pid`, `train_batches.host` and,
  inside SLURM, `slurm_job` (job id and restart count). The device column is the
  only record of which backend trained a trial.
- `bash scripts/train_batches_progress.sh [logs/train_batches/<timestamp>]`
  prints every trial of a run (default the latest): percentage from the trial's
  last `Steps: <done>/<total>` line, time left from the speed over its last 100
  such lines, and state (running, done, failed, queued). From another host it
  calls a run live only while `scontrol` reports its job RUNNING with the same
  restart count.
- A failed trial does not stop the rest; the script exits 1 if any failed. A
  trial whose `train.py` command line is already running is skipped, so two
  processes never share a results directory. It warns about processes holding
  ≥ 500 MiB of GPU memory, and when 4.5 GB per concurrent trial exceeds the
  available RAM (or `SLURM_MEM_PER_NODE`).
- **GPU memory with `N_PARALLEL > 1`:** unless a JAX memory variable is set, it
  exports `XLA_PYTHON_CLIENT_PREALLOCATE=false` (JAX otherwise reserves 75% of
  the GPU per process). Splitting the GPU with `XLA_PYTHON_CLIENT_MEM_FRACTION`
  failed: XLA's CUDA graphs and cuSolver handles need memory outside that pool,
  and trials died at init or minutes in.
- It exports `PYTHONUNBUFFERED=1`, or logs show nothing until exit.
- Ctrl-C (or TERM/HUP/QUIT) stops every running trial's process group (each
  trial runs in its own). Further signals are ignored during cleanup, and
  groups still alive after 10 s are SIGKILLed.
- ⚠ **It waits with `wait <pid>` and polls every 2 s, never `wait -n`.** Before
  bash 5.3, `wait -n` skips a job whose end the shell already reported and
  waits for the next one. In the final wait that was the never-ending progress
  monitor, so a finished run hung (seen on the cluster, 2026-10-28).
- Measured on the removed MJX groups (2026-10-24): a CPU trial on one
  performance core was as fast as a GPU trial sharing the GPU, and efficiency
  cores were about 2x slower. Not measured on rover.

## Plotting (`plotting/plot_training_stats.ipynb` + `plotting/config.yaml`)

`config.yaml` selects runs as `batches` × `experiments` × `trials` under
`base_path` (currently `/home/magraz/learned_reward/experiments/results`, an
uncommitted change). It is set to the rover groups and `rmappo` / `mappo_mlp`.
Cell 0 defines `load_reward_runs(batches)` and
`summarize_rewards(df, group_cols)`, which give the mean ± standard error of the
mean over trials at each exact `total_steps`. Both reward figures use them.
- **Single reward plot (cell 1):** every batch in `batches` on one axis.
- **Multi-batch reward grid (cell 2):** one subplot per batch in
  `multi_batches`, up to 3 columns, the same `experiments` and `trials` in each,
  one shared legend. Each experiment keeps one color across subplots, assigned
  by its position in `experiments`; `plot_colors` can override it by experiment
  name (a `batch/experiment` key has no effect here). Y axes are not shared. The
  cell prints a table of the trials found. Skipped if `multi_batches` is absent.
- **Reward by update (cell 3):** both figures again with the PPO update number
  on the x axis. `load_reward_runs` adds an `update` column (the row position:
  every stats row is one update), and `summarize_rewards(..., x="update")`
  groups on it, so seeds always align. Use it when arms spend different step
  budgets per update.
- Cells 4-7 plot `explained_variance`, `grouping_loss` (logged only by the
  removed torch `mappo`), `action_distribution` (discrete runs; it tries to import
  the removed `algorithms.hierarchical.skills` for labels and falls back to
  indices), and feudal goal-usefulness diagnostics. Series a run does not log
  are skipped.
- `custom_labels` and `plot_colors` still hold entries for removed experiments;
  they are only used when a name matches.
- `scripts/clean_notebooks.sh` clears the outputs of every notebook in the
  repository.

## Training stacks

Both stacks write the same stats pickles through `TrainingStatsTracker`
(`algorithms/mappo_jax/stats_tracker.py`; it lived in the removed
`mappo_vanilla` until 2026-10-30, so older runs have the same format):
`logs/training_stats_{checkpoint,finished}.pkl` with the same key set, so the
notebook reads any of them. `reward` is the
deterministic evaluation return; `total_steps`, `episodes`, timing series and
the per-update losses ride alongside. `TrainingStatsTracker.to_dict()`
left-pads any series shorter than `total_steps` with NaN (left, because a late
series' first element belongs to the iteration it first appeared in), so a
metric added mid-run, or missing from the checkpoint a run resumed from, stays
aligned.

**Truncation vs termination bootstrap.** Environments return
`terminated` (true end) and `truncated` (time limit) separately. Generalized
advantage estimation (GAE) needs `done = terminated | truncated` to cut the
advantage recursion, but only a termination should zero the value bootstrap.
So at a truncated step the collectors add `gamma * V(s_next)` to the stored
reward and keep `done` for the recursion. GAE's own bootstrap term is then 0
there, with no double count.
- `mappo_jax` resets finished envs in the same step, so it values the real
  `next_obs` **before** the reset.
- Without this, episodes that run to the time limit (every rover episode that
  does not use up all POIs) teach the critic that the final state is worth 0.
- `rmappo_jax` treats time limits as terminal by default
  (`use_proper_time_limits: false`, on-policy's default).

### JAX MAPPO (`algorithms/mappo_jax/`)

`algorithm=mappo_jax` (`AlgorithmEnum.MAPPO_JAX`) is a fully jitted MAPPO over
functional JAX environments: `reset(key) -> (obs, state)` and
`step(state, actions) -> (obs, state, reward, terminated, truncated, info)`,
obs `(n_agents, obs_dim)`, `info["task_reward"]` the team reward, no
auto-reset. It was written as a logic mirror of the removed `mappo_vanilla`
(torch), so runs from the old project are comparable. Its only environment now
is the rover domain.

- **`run.make_env(env_config)`** has one branch per environment, each passing
  its constructor arguments explicitly; an unknown `environment` raises. The
  `ROVER` branch hands the whole `env:` block to `make_rover_env`, which
  validates it. `rmappo_jax` builds envs through the same function.
- **Cadence** (the removed vanilla `VecMAPPOTrainer.train`'s): jitted `collect_fn` resets all envs
  at the top of every rollout, scans `params.n_steps` (per-update batch
  `n_steps * n_envs`; default 1048 × 32 = 33,536), restarts envs that finish
  mid-rollout (`trainer.restart_done_envs`), and bootstraps the final value.
  Then jitted `update_fn` and, every 10 updates plus the last, the deterministic
  `eval_fn` (`n_eval_episodes`, default 32). `reward` carries the last
  evaluation forward in between. ⚠ Because every rollout starts from a reset, a
  rollout shorter than `max_steps` never trains on the back half of an episode.
- **PPO update** (`mappo.ppo_update` = `prepare_advantages` →
  `flatten_samples` → `optimize_samples`): GAE on the scalar team reward with
  the shared critic, per-env-stream advantage normalization (unbiased std),
  timestep-centric minibatches (critic once per timestep), loss
  `policy + val_coef*value + ent_coef*entropy` with separate Adams for actor and
  critic, pre-update `explained_variance`. Deviations from the old vanilla
  stack: the trailing
  partial minibatch is dropped (jit needs static shapes), and only the shared
  actor is implemented (`parameter_sharing: false` raises).
- **Networks** (`network.py`, flax): 2-layer Tanh MLPs with orthogonal init,
  actor hidden `hidden_dim` (168), critic hidden `2*hidden_dim`, learned
  state-independent `log_action_std` (init −0.5, clamped to [−5, 2]).
  Hand-rolled diagonal Gaussian and categorical distributions. Continuous and
  discrete action spaces (`env.discrete`).
- **Centralized critic input**: `trainer.global_state_fn(env)` /
  `global_state_dim(env)`, the one definition shared by training, checkpoint
  reload and `view()`. An env may publish `global_state(state)` +
  `global_state_dim`; otherwise it is the concatenation of observations (rover:
  `16 * n_agents`). Both runners print `centralized input: gs_dim=...` at launch.
- **Optional env hooks**, decided at trace time so an env without them takes
  the plain path. Rover has none.
  - `avail_actions(state)`: legal-action masks. Fetched from the pre-step state
    and stored in `Transition.action_mask`; envs without it store a scalar
    placeholder, and `ppo_update` switches on `action_mask.ndim == 4`.
  - `info["active"]`: per-agent mask removing inactive agents from the losses;
    defaults to all-ones.
  - `per_agent_rewards(env)`: whether the env's agents can be paid differently
    (an env's `per_agent_rewards` attribute, else
    `reward_mode == "difference_rewards"`). True switches the critic to a
    per-agent value head and GAE to the agent axis; rover is true under
    `difference_rewards` only. `step` may return one reward per agent even when
    false (rover's `shared_reward`): `trainer.training_reward` then trains on
    agent 0's column, which is the team reward, with one team value. Both
    stacks call it. `Transition.team_reward` always carries the team scalar for
    logging and evaluation.
- **Outputs**: params as flax msgpack (`models/models_{checkpoint,finished}.msgpack`).
  **Resume** (`checkpoint=true`): the stats checkpoint restores the progress
  counters, and `models/train_checkpoint.msgpack` restores actor/critic params,
  optimizer states, step counters and both random-number-generator (RNG)
  key chains (training and evaluation). Both are saved at every
  log point and at finish, so re-running with a larger `n_total_steps` extends
  a finished run. `evaluate=true` prints the mean evaluation return.
- **`view=true`** is implemented for rover only (`_view_rover`; any other env
  raises `NotImplementedError`). It writes `logs/rover/arena.png`
  (`plot_arena`) and three `trajectory_seed<k>.png` (`plot_trajectory`) for
  seeds `rng_seed + 1..3`. Episodes run through
  `rover.rollout.rollout_view_policy`, a host loop that resets the policy each
  episode and stops at termination or truncation. `README.md` has the command.
- **Subclass hooks** (used by `rmappo_jax`): `_init_trial` (directories and
  seeding), `_make_train` (default `make_train(self.config, self.env)`),
  `_steps_per_update`, `_eval_every`, `_view_policy` (returns an object with
  `reset()` and `__call__(obs, mask=None)`; `StatelessViewPolicy` by default),
  and in `trainer.py` `restart_done_envs` and
  `make_eval_fn(env, n_episodes, EvalPolicy(init, act))`. `train()` treats the
  trajectory and bootstrap value opaquely.
- `conf/algorithm/mappo_jax.yaml`: `n_epochs` 6, `n_minibatches` 8, `n_steps`
  1048, lr 3e-4, γ 0.99, λ 0.95, clip 0.2, entropy 0.01, grad clip 0.5,
  `n_total_steps` 1e8, `hidden_dim` 168. Model group `mlp` (`hidden_dim: 168`,
  restored 2026-10-30).
  ```
  uv run python train.py algorithm=mappo_jax env=rover_2poi_c2 model=mlp trial_id=0
  ```
- **Smoke-verified 2026-10-30** on `rover_2poi_c2` (CPU, `trial_id=smoke`):
  one 33,536-step update in 1.9 s including compilation, and `view=true`. **No
  training result.** Again after the reward modes (2026-10-30, 2 updates each,
  `mlp` from a scratch search path): `shared_reward` saves a critic with 1
  output, `difference_rewards` one with 4 (per rover). `rmappo_jax` trains in
  both modes, and an existing `rmappo` run (`rover_2poi_c2` trial 0) still
  loads and evaluates to 2.00, its logged final reward.

### Reference MAPPO (`algorithms/rmappo_jax/`, added 2026-10-27)

`algorithm=rmappo_jax` (`AlgorithmEnum.RMAPPO_JAX`) reimplements the reference
MAPPO code in JAX over the same functional envs. The reference is
marlbenchmark/on-policy (`main`, `onpolicy/algorithms/r_mappo`; Yu et al. 2022,
"The Surprising Effectiveness of PPO in Cooperative Multi-Agent Games"), ported
from its source files (read 2026-10-27). `mappo_jax` is plain PPO with a
centralized critic; it lacks value normalization, Huber and
clipped value losses, batch-wide advantage normalization, persistent rollouts
and the GRU (gated recurrent unit). This stack is the one to cite as MAPPO.

```
uv run python train.py algorithm=rmappo_jax env=rover_2poi_c2 model=rmappo trial_id=0     # GRU (on-policy's rmappo)
uv run python train.py algorithm=rmappo_jax env=rover_2poi_c2 model=mappo_mlp trial_id=0  # feedforward (on-policy's mappo)
```

- **Defaults are on-policy's `config.py`** (`conf/algorithm/rmappo_jax.yaml`):
  - networks: `hidden_size` 64, `layer_N` 1, ReLU, LayerNorm after every layer
    and on the input;
  - optimizers: lr and critic_lr 5e-4, Adam ε 1e-5, gradient-norm clip 10;
  - PPO: 15 epochs × 1 minibatch, clip 0.2, entropy 0.01, value coefficient 1,
    γ 0.99, λ 0.95;
  - ValueNorm, Huber (δ 10), clipped value loss, active masks in both losses;
  - `n_steps` (episode_length) 200, GRU chunks of 10 steps;
  - `n_total_steps` **1e7** (on-policy's `num_env_steps`). ⚠ The yaml comment
    beside it says "the repo's 1e8"; the value is 1e7.
- **Key names.** on-policy's names are kept except where the repo's other
  algorithm groups already have a key: `n_steps` (= `episode_length`),
  `n_epochs` (= `ppo_epoch`), `n_minibatches` (= `num_mini_batch`), `eps_clip`
  (= `clip_param`), `ent_coef`, `val_coef`, `grad_clip` (= `max_grad_norm`),
  `lmbda`, `n_eval_episodes`. The full map is in `types.py`.
- **Model groups `rmappo` / `mappo_mlp`** set only `use_recurrent_policy`, which
  has no default, so a model group without it raises a `TypeError` before any
  results directory exists.

**Reproduced from the source:**
- **Persistent rollouts.** Envs are reset once. Env states, observations, both
  GRU hidden states and the episode mask carry over between updates in
  `RunnerState`. A finished env restarts at once; its hidden states are zeroed
  and its next mask is 0.
- **Critic input**: the centralized state (`mappo_jax.trainer.global_state_fn`;
  the concatenated observations for rover, as on-policy's MPE runner builds).
  Every agent receives the team reward, or its own reward when
  `per_agent_rewards(env)` (on-policy's buffer holds one reward per agent;
  through `mappo_jax.trainer.training_reward`).
  - **One read per env** (2026-10-30, not on-policy's, see Deliberate
    deviations): `trainer.critic_step` runs the critic once on each env's
    state, with one GRU hidden state per env (`critic_h`
    `(n_envs, recurrent_N, H)`). Its head (`trainer.n_critic_outputs`) emits
    one team value, kernel `(hidden, 1)`, or one value per agent under
    `per_agent_rewards(env)` (rover `difference_rewards`), kernel
    `(hidden, n_agents)`. A team value is broadcast to every agent, so
    values, GAE, the loss and ValueNorm stay per (env, agent) row as in
    on-policy, active masks included.
  - The update trains the critic on per-env chunks
    (`mappo.gather_critic_minibatch`, its own permutation from
    `fold_in(epoch key, 1)`, split into the same `n_minibatches`); the actor
    keeps its (env, agent) chunks (`mappo.gather_minibatch`).
- **GAE on denormalized values**, per agent (`mappo_jax.mappo.compute_gae`,
  whose `1 - done[t]` is on-policy's `masks[t + 1]`).
- **Advantage normalization** is batch-wide: mean and population std over the
  active rows of the whole (time × env × agent) batch, `+ 1e-5` on the std.
- **Samplers** (`mappo.gather_minibatch`, the actor's; the critic's are per
  env, see Critic input): recurrent = each (env, agent)
  trajectory cut into `data_chunk_length` chunks in on-policy's order, each
  starting from the hidden state stored at its first step; feedforward = chunks
  of length 1. A fresh permutation every epoch, remainder dropped. Inside a
  chunk a `lax.scan` applies `h ← h · mask_t` at every step (tested against
  on-policy's `RNNLayer`).
- **Losses**: actor = PPO-clip minus `ent_coef ×` entropy over active rows;
  critic = `max(huber(e), huber(e_clipped))` scaled by `val_coef`, prediction
  clipped within `eps_clip` of the stored normalized value. Separate Adams and
  gradient-norm clips; logged gradient norms are pre-clip.
- **ValueNorm** (`value_norm.py`: β 0.99999, ε 1e-5, variance floor 1e-2)
  updated from every minibatch's returns, inactive rows included, before that
  minibatch's critic loss. Before the first update `denormalize(v) = 0.1 v`.
- **Networks** (`network.py`): orthogonal init at the activation's gain, zero
  biases, LayerNorm ε 1e-5; the GRU uses torch's layout; action head gain 0.01;
  the continuous head's learned `log_std` starts at 0 (std 1) with no clamp;
  unavailable discrete actions get logit −1e10; critic head gain 1.
- **Time limits are terminal** by default (`use_proper_time_limits: false`).
  With `true`, this repo's bootstrap is used instead (the real successor valued
  through the critic's advanced hidden state, `γ · denormalize(V)` added to the
  reward). That is not on-policy's own `bad_masks` path.
- **`use_linear_lr_decay`**: `lr · (1 − update / num_updates)`, constant within
  an update.

**Deliberate deviations:**
- The critic reads each env's centralized state once, with one hidden state
  per env, emitting a team value or one per agent (see Critic input).
  on-policy repeats the state for every agent row, which gives every agent the
  same value and so cannot serve per-agent rewards. Under a shared reward the
  two compute the same function; the random draws, the critic's minibatches
  and its chunk order differ, so runs before 2026-10-30 (every rover result on
  disk) are not reproduced exactly.
- `reward` is this repo's deterministic evaluation (`n_eval_episodes` 32, every
  `eval_interval` = 25 updates). It does not affect training.
- Under the GRU `n_steps` is rounded up to a multiple of `data_chunk_length`
  with a printed notice (on-policy silently builds chunks spanning two
  trajectories). `validate` refuses an unrounded config.
- A `checkpoint=true` resume restores the networks, both Adam states, ValueNorm
  and the RNG chains, but restarts episodes and hidden states. The checkpoint
  stores `config_meta` (the shape-free settings in `run.META_FIELDS`); resuming
  under different values raises and names them.
- Refused: `parameter_sharing: false`.
- Not ported: PopArt (adaptive value-target normalization), the naive
  recurrent policy, the convolutional (CNN) base, stacked frames,
  other action spaces, `weight_decay`, Xavier init, SMAC-specific options.

**Before comparing with `mappo_jax`:** a gap mixes hyperparameters with
implementation. On groups that do not set `n_steps` (all rover groups) the
batch is 200 × 32 = 6,400 env steps against `mappo_jax`'s 1048 × 32 = 33,536,
i.e. ~5× the updates, each of 15 full-batch epochs. To separate the two, add a
model group with `mappo_jax`'s hyperparameters (in a model group, never on the
command line, because of the results path).

⚠ **Memory:** before 2026-10-30 the critic input was repeated for every
agent row, and with on-policy's single minibatch a group with many agents and a
wide concatenated state could fill a GPU (measured 12.3 GB at 16 agents with a
640-wide state on a removed MJX group; 5.4 GB with `n_minibatches: 4`). The
critic now reads it once per env; not re-measured. Rover's state is
`16 * n_agents` wide (256 at 16 rovers).

**Checks:** `uv run pytest algorithms/tests/test_rmappo_jax.py -q` (36 tests,
CPU stubs). They cover the GRU against
torch's `nn.GRU` through on-policy's `RNNLayer`, ValueNorm against a numpy
transcription, Huber/mse, the clipped value `max`, inactive rows, batch-wide
normalization, recurrent re-evaluation reproducing the rollout's log-probs to
1e-5 (with a negative control), a first-minibatch ratio of exactly 1,
persistent rollouts, both time-limit modes, no bootstrap after termination,
masked actions, end-to-end runs, `n_steps` rounding, the defaults, the
rejections, the decay schedule, the checkpoint round trip, each agent's reward
in the buffer (shared and per agent), and the critic's single read per env
(head width 1 or `n_agents`, one hidden state per env, a team value identical
across agents, an update that moves it). Re-evaluation covers the critic's
per-env chunks for a team head and a per-agent head, and the masks negative
control breaks both the actor's log-probs and the critic's values.
`algorithms/tests/test_rover_trained_view.py` (3 tests) checks that
`rollout_view_policy` + `RecurrentViewPolicy` reproduce an independent
functional rollout (both model types), reset the GRU between episodes, and stop
at early termination.

**Smoke-verified 2026-10-27** on `rover_2poi_c2` (`trial_id=smoke`): train,
resume and evaluate; eval 2.00 of 2 by 128k steps; 0.02 s collection / 0.09 s
update per 6,400-step update; 1.1 GB GPU peak. (The other rows of that table
were removed MJX and SMAX groups.) **No full training result.**

## Rover domain (`environments/rover/`, added 2026-10-27)

A pure-JAX, physics-free 2D multi-rover world. Rovers steer; points of interest
(POIs) pay the team while enough rovers are near them.
- **Trains with** `algorithm=rmappo_jax` (`model=rmappo` / `mappo_mlp`; the
  default config) or `algorithm=mappo_jax` (`model=mlp`).
  - The base group sets a literal `n_envs: 32`, inherited by every variant.
    `make_rover_env` accepts it as a runner key (`config._RUNNER_KEYS`) and
    ignores it.
  - What each rover trains on is `env.reward.mode` (see Default reward):
    `shared_reward` (default; `per_agent_rewards(env)` false, so both stacks'
    critics emit one team value) or `difference_rewards` (true: one value per
    rover). Both critics read the state once per env. Both runners print
    `reward_mode=`.
  - ⚠ The mode is not in the results path
    (`<env group>/<model group>/<trial>`). `env.reward.mode=...` on the command
    line trains into the shared-reward runs' directories, and with
    `checkpoint=true` it resumes them. Give each mode its own env group.
  - No global-state hook: the critic reads the concatenated observations
    (64 dims at 4 rovers).
  - `view=true` writes the arena and trajectory PNGs (see JAX MAPPO).
  - **Smoke result (2026-10-27, 128k steps = 20 updates):** `rmappo` reached
    deterministic eval 2.00 of 2.00 (0.01 untrained, 1.81 at update 10).
    Replayed over 256 episodes: mean return 1.998, both POIs used up in 255.
    Scripted coalition 2.000, random 0.016. The learned policy is a sequential
    swarm (all four rovers on one POI, then two cross to the other), finishing
    in a median of 70 steps against the scripted split's 28.
- Outside training: `environments.rover.config.load_rover_env(variant)`.
  ```
  from environments.rover.config import load_rover_env
  env = load_rover_env("rover_2poi_c2", overrides=["env.n_agents=6"])
  ```
- **Contract** (the functional API above, serving evolutionary algorithms and
  RL alike): `step` takes no PRNG key (dynamics are deterministic). Attributes
  `n_agents`, `n_pois`, `observation_dim`, `action_dim = 2`, `discrete = False`,
  `max_steps`, `world_size`, `world_diagonal`, `spawn_region`,
  `per_agent_rewards`, `reward_mode`. `step`'s reward has shape
  `(n_agents,)` in every mode; `info["task_reward"]` is the team reward `G`,
  which evaluation, logging and rollouts score. `info` carries
  `task_reward`, `agent_positions`, `agent_headings`, `agent_speeds`,
  `poi_remaining`, plus the reward model's keys (`shared_reward`,
  `difference_rewards`, `poi_observed`, `poi_unlocked`, `poi_active`,
  `poi_payout`). State (`state.py`) is a flat
  pytree of arrays: `RoverEnvState(rovers=RoverState(position, heading),
  pois=POIState(position, coupling, radius, value, rate, remaining, order,
  hide_until_unlocked), t)`.
- **Dynamics** (`BaseRoverEnv.move`). Action `a = [turn, speed]` in `[-1, 1]²`:
  - `heading <- wrap(heading + a0 * max_turn)`, wrapped to [-π, π);
  - `speed = (a1 + 1)/2 * max_speed`, set directly each step (not integrated),
    so `a1 = -1` stops the rover;
  - the rover turns first, then moves along its new heading; its position is
    clipped to `[0, W] × [0, H]`.

  Walls are inert and never end the episode. Spawn: uniform in `spawn_region`
  with a uniform heading.
- **Observation** (`SectorDensityObservation`, 16 dims): 8 agent sectors, then
  8 POI sectors.
  - **Sectors are heading-relative.** Sector 0 is centred on the heading and
    spans [-22.5°, 22.5°); numbering runs counterclockwise (sector 2 left,
    sector 6 right). There is no heading channel, so world-frame sectors would
    leave a turning policy unable to tell which way to turn.
  - **Presence** = `Σ over entities in the sector with d < R of (1 − d/R)`,
    divided by `n_agents − 1` (agents) or `n_pois` (POIs); bounded in [0, 1].
  - The agent block excludes self **by index**, so a co-located teammate is
    sensed.
  - Used-up POIs are not sensed but still count in the `n_pois` denominator.
  - Locked POIs (see Temporal coupling) are sensed like unlocked ones, so the
    observation does not say which POIs can pay yet, unless the POI sets
    `hide_until_unlocked` (see Temporal coupling). The rule is
    `pois.visible(pois)`.
  - `sensor_radius: null` = the arena diagonal (global sensing).
  - Binning goes through `environments/sectors.py:sector_onehot`.
- **Default reward** (`CouplingDepletionReward`), on post-move positions:
  - a POI is active when ≥ `coupling` rovers are within its radius (inclusive),
    its `remaining` budget is > 0, and it is unlocked (see Temporal coupling);
  - an active POI pays `min(rate, remaining)` per step and its budget drops by
    that much; extra rovers add nothing;
  - budgets below 1e-6 snap to 0, so value 1.0 / rate 0.05 depletes in exactly
    20 coupled steps;
  - the team reward `G` is the sum of payouts (`RewardOutput.team_reward`,
    `info["task_reward"]`). `info["shared_reward"]` is `G` for every rover; it
    replaced `info["agent_rewards"]` (each payout split equally among the
    rovers observing it) on 2026-10-30;
  - `CouplingDepletionReward(mode)` (yaml `reward.mode`, default
    `shared_reward`) picks `step`'s per-rover reward by its info key:
    `shared_reward` or `difference_rewards`. Both are in `info` in either
    mode. It sets `per_agent_rewards = (mode != "shared_reward")`; an unknown
    mode raises;
  - `info["difference_rewards"]` (added 2026-10-30) is each rover's
    difference reward `D_i = G − G_{−i}`: this step's team reward minus the one
    with rover i removed, the other rovers and the POI budgets unchanged.
    - Removing rover i lowers the count of every POI it observes by one, so
      `D_i` is the payout of the POIs whose coupling it completes exactly.
      Computed as a batch of counterfactual counts through the same
      `_payout` rule as `G`.
    - A rover beyond a POI's coupling earns 0 from it (where the equal split
      still paid it); so does a rover waiting alone at a POI whose coupling is
      not met. The `D_i` need not sum to `G`.
    - It is per step: the counterfactual does not replay how the missing rover
      would have changed later budgets. It keeps the locks too, so a rover at
      a locked POI earns 0 and no rover is credited for unlocking later POIs;
  - `terminated` = every POI used up; `truncated = (t ≥ max_steps) &
    ~terminated`, so the two never fire together.
- **Temporal coupling (POI `order`, added 2026-11-02).** Every POI has an
  integer consumption `order` (`POIState.order`, int32), next to its spatial
  `coupling`. It pays only once every POI of a lower order is used up; equal
  orders constrain nothing, and gaps in the numbering do not matter.
  - Set per entry in `pois.entries` (`order:`, integer ≥ 0, default 0), so
    every existing variant (all 0) is unchanged. It is a key of the `static`
    POI model, not a separate `type:`: the pytree state cannot hold `None`, so
    every POI carries the field anyway. The `_ord` groups use it.
  - `pois.unlocked(pois)` is the one definition of the rule, read by the
    reward and by the scripted coalition policy. The lowest order with budget
    left is always unlocked.
  - The lock is read from the budgets at the start of the step: a POI unlocks
    on the step after the last earlier POI's final payout.
  - `StaticPOIs` warns when a POI is ordered after one that can never pay
    (coupling > `n_agents`), since it never unlocks either.
  - **Hiding locked POIs (`hide_until_unlocked`, added 2026-11-03).** A POI
    with `hide_until_unlocked: true` is not sensed by the rovers while it is
    locked. It appears on the step it unlocks.
    - What changes: the observation only, through `pois.visible(pois)`:
      `remaining > 0 & (~hide_until_unlocked | unlocked)`. The reward and the
      lock are unchanged: a flagged POI pays exactly when an unflagged one
      would.
    - The rover critic reads the concatenated observations, so it does not
      see a hidden POI either.
    - A hidden POI still counts in the `n_pois` normalizer, as a used-up one
      does: while locked it reads exactly as used up, and once unlocked
      exactly as an unflagged POI.
    - Set per entry in `pois.entries`, or as a `pois:` block default (like
      `radius` / `value` / `rate`) that an entry overrides; the base group
      sets `hide_until_unlocked: false`. One line hides every locked POI of
      a variant: `pois: {hide_until_unlocked: true}`, or on the command line
      `env.pois.hide_until_unlocked=true`.
    - A POI of the lowest order is never locked, so its flag does nothing.
      `StaticPOIs` warns when the flag is set but acts on no POI (e.g. on a
      layout with a single order), and raises on a non-boolean value.
    - It is not in the results path, so a command-line override trains into
      the unflagged group's directory: give a flagged layout its own env group.
    - **`<name>_ord_hidden` groups (2026-11-03)**, one per `_ord` group:
      `rover_4a_2poi_c2_ord_hidden`, `rover_4a_4poi_c1234_ord_hidden`,
      `rover_6a_6poi_c3_ord_hidden`, `rover_16a_2poi_c8_ord_hidden`,
      `rover_16a_6poi_c844844_ord_hidden`. Each inherits its `_ord` group and
      sets only `pois.hide_until_unlocked: true`; each composes to exactly its
      parent's config plus that key, builds without warnings, and hides at
      reset every POI above the lowest order.
    - ⚠ **Where hiding acts.** Over 8 scripted-team episodes per group, the
      observation differed from the parent's on 129 steps
      (`rover_4a_2poi_c2_ord_hidden`, global sensing) and 494
      (`rover_4a_4poi_c1234_ord_hidden`, global sensing), and on **none** in
      `rover_16a_2poi_c8`, `rover_6a_6poi_c3` and `rover_16a_6poi_c844844`
      (`_ord_hidden`). Their sensor radius (2.0, 2.0, 4.0) already keeps a
      locked POI out of range: it is farther than that from the spawn box
      and from the POI being worked. There, hiding only acts when a rover
      strays within range of a locked POI, so an `_ord` vs `_ord_hidden`
      comparison on those layouts measures little.
    - Smoke-verified 2026-11-03: `rmappo_jax` trains (2 updates, CPU) and
      evaluates on `rover_4a_2poi_c2_ord01` (now `rover_4a_2poi_c2_ord`) with
      the flag on. Every variant builds with the flag off and on, observations
      stay in [0, 1], and POIs are hidden only in the 5 `_ord` variants.
  - The figures add `order n` to each POI's label when a layout has more than
    one order.
- **Pluggable components** (`components.py`): `BaseRoverEnv(n_agents, pois,
  observation=None, reward=None, ...)` takes any
  - `POIModel`: `init(key, env)`; `step(pois, rovers, t)` (identity by default,
    the hook for moving POIs); `validate(env)`;
  - `ObservationModel`: `obs_dim` and `__call__(env, state)`;
  - `RewardModel`: `__call__(env, rovers, pois) -> RewardOutput(reward,
    team_reward, pois, terminated, info)`: `reward` `(n_agents,)` is what the
    rovers train on, `team_reward` `()` what evaluation scores (kept apart
    because difference rewards do not sum to `G`). It returns the updated
    `POIState`, since depletion is part of the reward's semantics. Class
    attribute `per_agent_rewards` (default true; false = every rover is
    always paid the same).

  `__init__` checks the declared `obs_dim` against the emitted width, and the
  reward shapes, with `jax.eval_shape`. Register a new implementation in `config.py`
  (`POI_MODELS` / `OBSERVATIONS` / `REWARDS`, keyed by the block's `type:`) to
  make it selectable from yaml.
- **Config: one Hydra env group per POI layout (variant).**
  `conf/env/rover_2poi_c2.yaml` is the default: 4 rovers, 10×10 arena, spawn in
  `[4,6]²`, 100 steps, `max_speed` 0.5, `max_turn` π/4, global sensing, POIs at
  (2,8) and (8,2) with coupling 2, radius 1, value 1, rate 0.05. Variants
  inherit it (`defaults: [rover_2poi_c2, _self_]`) and override what differs; a
  `pois.entries` override replaces the whole list.

  | variant | rovers | arena | sensor radius | POIs (coupling) |
  |---|---|---|---|---|
  | `rover_2poi_c2` | 4 | 10×10 | global | (2,8) 2, (8,2) 2 |
  | `rover_2poi_c8` | 16 | 10×10 | 2.0 | (2,8) 8, (8,2) 8 |
  | `rover_4poi_c1234` | 4 | 10×10 | 2.0 | 4 corners: 1, 2, 3, 4 |
  | `rover_6poi_c3_ring` | 6 | 10×10 | 2.0 | 6 of coupling 3 on a radius-3.5 ring |
  | `rover_6poi_c8-4-4-8-4-4` | 16 | 20×20, spawn `[8,12]²` | 4.0 | (10,12) 8, (15,17) 4, (5,17) 4, (10,8) 8, (15,3) 4, (5,3) 4 |
  | `rover_4a_3poi_c3_implicit012` | 4 | 15×15, spawn `[3.42,3.92]×[4.35,4.85]` | global | (1,1) 3, (7.75,4) 3, (13,13) 3 |

  **`rover_4a_3poi_c3_implicit012`** (2026-11-03): a visiting order set by
  layout and values, with no `order` key. Only the depletion order 0 → 1 → 2
  (the corner behind the spawn, then the POI nearest the spawn, then the far
  corner) collects the maximum 6.0 within 100 steps. Heading for the nearest
  POI first is the trap.
  - Why: coupling 3 is more than half of the 4 rovers, so only one POI pays per
    step. The paying steps, `Σ value / rate` = 12 + 24 + 24 = 60 (values 1.2,
    2.4, 2.4 at rate 0.1), are the same in every order; only travel differs,
    and 40 steps are left for it. With one team and static POIs the values can
    only set this budget; the order itself is a shortest-path question. The
    nearest POI can come second but never uniquely last (starting there and
    reversing the path is never longer).
  - Checked over 1000 spawn seeds: the scripted route 0-1-2 (`ROUTE_PLANS`
    has it; `scripted_steps --variant rover_4a_3poi_c3_implicit012`) collects
    6.0 on every seed, in 93-98 steps, with 33-38 non-paying steps. A lower
    bound for any policy (heading-limited entry, then straight-line travel
    between disc edges, over every walk that uses the POIs up in that order)
    puts every other order at ≥ 44 non-paying steps. The scripted routes of the
    wrong orders collect 61-89% of the maximum (nearest-first 88.7%). The
    total paying steps must stay within 57-62 for the order to stay unique.
    The check script is not in the repository.
  - The same shape in 10×10 leaves wrong orders 77-92% of the maximum and
    0-3 spare steps; in 20×20, 59-78% but a much harder first coupling.

  `rover_2poi_c8_diffrew` (2026-10-30) is `rover_2poi_c8` with
  `reward.mode: difference_rewards`, its own results path. Smoke-verified
  under `rmappo_jax` (2 updates, CPU): critic head `(64, 16)`, reload through
  `evaluate=true`.

  `load_env_config` composes `conf/config.yaml` with `env=<variant>`, the same
  path as `train.py`. `make_rover_env` raises on
  unknown `env:` keys, unknown `dynamics` keys, unknown component types and
  unknown POI entry keys. `StaticPOIs` raises on a POI outside the arena or
  `coupling < 1`, and warns when `coupling > n_agents`.
- **Rollouts** (`rollout.py`):
  - `rollout(env, policy_fn, key, params=None, n_steps=None)` is a `lax.scan`
    over `max_steps` (or `n_steps`). The state is frozen after the episode ends
    and later rewards are 0. Returns a `Trajectory` (positions, headings,
    `poi_remaining`, actions, rewards, `done`, `episode_return`; `rewards` is
    the team reward `info["task_reward"]`, not `step`'s per-rover reward) and
    vmaps over
    keys or over policy parameters (evolutionary fitness).
    `policy_fn(params, key, obs, state)`.
  - `rollout_reset(env, key)` returns the initial state `rollout` starts from,
    so a planner can read it and pass its plan as `params`.
  - `rollout_view_policy(env, policy, key)`: a host loop (not jittable) for a
    runner's stateful view policy; used by `view()`.
- **Demonstration policies** (`policies.py`, the scripted ones read the
  privileged state):
  - `random_policy`;
  - `make_scripted_coalition_policy(env)`: rovers fill available (undepleted
    and unlocked) POIs in index order, `coupling` rovers each, and retarget
    when a POI is used up;
  - `make_route_policy(env)` (2026-10-30): `params` is an `(A, K)` array of
    per-rover POI routes; a rover's target is the first POI on its route with
    budget left. It drives radially to the nearest point at 0.9 × radius
    (`PARK_DEPTH`), then slides to the point at that depth facing its next POI
    and turns in place toward it. It never leaves a radius whose POI still has
    budget.
  - `plan_routes(env, state, teams)` assigns rovers to the routes of a team plan
    (`ROUTE_PLANS[variant]`: `(route, n_rovers)` pairs), minimizing the latest
    entry into a first POI, then the distance to the parking point. Entry times
    come from `entry_steps`, which simulates the controller, so they are exact.
    The assignment runs on the host (scipy). `planned_rollouts(env, keys,
    teams)` plans and runs a batch of seeds.
  - The scripted policies share the steering helper `_steer`.

  Measured (8 seeds): the scripted coalition collects 2.0/2 in 26–29 steps on
  the default, 6.0/6 in 89–92 on the ring, and 3.75–4.0/4 on `c1234`, where the
  coupling-4 POI is often still paying at step 100. Random rovers collect 0 on
  the coupling-2 and coupling-3 layouts.
- **Steps to solve `rover_6poi_c8-4-4-8-4-4` (measured 2026-10-30, seeds
  0–999).** Solved = every POI used up (return 6.0).
  ```
  uv run python -m environments.rover.scripted_steps [--n-seeds 1000] [--figures 2]
  ```
  - **The plan.** Couplings sum to 32 = 2 × 16 rovers, so two waves: 8 rovers
    on each coupling-8 POI, (10,12) and (10,8), about 2 units from the spawn
    box; then 4 rovers per corner, each leaving from the nearer centre POI.
  - **Route policy:** solves 1000 of 1000 seeds in 52–56 steps (median 54,
    mean 53.71). The index-order scripted policy solves 937 of 1000 in 80–88.
  - **Step count** = the step the last centre-team rover gets inside (3–7),
    plus 49 on every seed: 20 coupled steps at the centre POIs, an 11-step leg
    (disk gap 5.07 units at 0.5 per step), 20 coupled steps at the corners.
  - **How close to optimal.** Two-wave lower bound = 49 + the best 8/8 split's
    latest entry, each entry bounded with turning included (the heading turns
    at most π/4 per step, so step t covers at most
    0.5 · cos(max(0, |φ₀| − tπ/4)) along the initial bearing). The policy meets
    this bound on 736 of 1000 seeds and is 1 step above on the other 264.
    Corners first needs ≥ 59 steps. Any policy needs ≥ 50 (the 640 rover-steps
    the POIs need could not otherwise be supplied). The two-wave bound is not
    proven for every policy.
- **Figures** (`visualize.py`, matplotlib, dataviz reference palette):
  `uv run python -m environments.rover.visualize [--variants ...] [--seeds 0 1]
  [--policies random scripted] [--no-trajectories]` writes
  - `plotting/rover/arenas/<variant>.png` for every `conf/env/rover_*.yaml`:
    the arena, each POI's radius disc with its coupling inside and value / rate
    below, and one rover's sensing radius at the spawn-region centre
    (`draw_sensing_radius`; with global sensing only a centre dot and a "senses
    the whole arena" label). This is also `view()`'s `arena.png`;
  - `plotting/rover/trajectories/<variant>_<policy>_seed<k>.png`: rover paths
    (start dot, arrow at the final heading), used-up POIs hollow with their
    depletion step, payout steps, cumulative team reward.

  Rover colours follow the categorical order; above 8 rovers all paths share
  one neutral colour. POIs are neutral ink.
- `environments/rover/mpe_jax/mpe_introduction.py` is a standalone JaxMARL MPE
  (multi-agent particle environment) demo with random actions; it is not part
  of the rover domain and nothing imports it. It is the only remaining user of
  the `jaxmarl` package.
- **Checks:** `uv run pytest algorithms/tests/test_rover_env.py -q` (56 tests,
  CPU; 27 fail on stale variant names, see Loose ends). Hiding: the flag's
  default and per-entry override; a flagged POI absent until it unlocks, then
  sensed exactly as an unflagged one; over a jitted episode of
  `rover_4a_2poi_c2_ord`, a flagged POI reads as used up while locked and
  trajectories and rewards are unchanged; a non-boolean value rejected; the
  no-effect warning. Temporal coupling: orders default to 0, `unlocked` with ties and gaps,
  a later POI paying from the step after the earlier one is used up, a locked
  POI paying no one (difference rewards included), the `G − G_{−i}` check
  repeated with orders, the scripted team working ordered POIs in order (2
  rovers; ignoring the lock scores 0), bad orders rejected, the never-unlocks
  warning, and the figures' order labels. The rest cover shapes, jit and vmap; the sector frame and ±22.5° boundaries; exact
  presence values; co-located teammates; invisible used-up POIs; observations in
  [0, 1] over random rollouts of every variant; turn/speed mapping, heading
  wrap, wall clip; coupling and the inclusive radius; the shared reward;
  both reward modes (`step`'s reward = `info[mode]`, team reward `G`), a
  scored rollout under `difference_rewards` (2.0, not `Σ D_i` = 4.0), a
  reward model without one reward per rover rejected, an unknown mode;
  difference rewards (hand-worked cases, and `G − G_{−i}` recomputed with each
  rover deleted over 64 random layouts of `rover_4poi_c1234`);
  depletion after exactly value/rate steps; termination needing every POI;
  truncation never firing with termination; custom components and a wrong
  `obs_dim` being rejected; variant composition (dynamics and observation type
  inherited, `sensor_radius` free); Hydra overrides; the
  bad-config errors; the scripted team; the frozen rollout; population vmap;
  determinism; the route policy on `rover_6poi_c8-4-4-8-4-4` (return 6.0 in
  exactly last centre entry + 49 steps, no rover leaving a POI that needs it,
  plan validation); the PNG writers for every variant.
