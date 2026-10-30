from pathlib import Path

from algorithms.types import AlgorithmEnum
import jax


def _dispatch(
    algorithm: str,
    exp_dict: dict,
    env_config: dict,
    batch_dir: Path,
    results_dir: Path,
    trial_id: str,
    view: bool = False,
    checkpoint: bool = False,
    evaluate: bool = False,
    debug: bool = False,
):
    """Build the per-algorithm Experiment + Runner and run it.

    The single dispatch tail behind the Hydra entry point (`train.py`).
    `batch_dir` is only used by the runners for `combined_affinities` checkpoint
    resolution (`batch_dir.parents[1]/results`); `results_dir` is the runner's
    `trials_dir` (`results/<batch>/<name>`), under which it writes
    `<trial_id>/{logs,models,videos}`.
    """

    match (algorithm):

        case AlgorithmEnum.MAPPO:
            from algorithms.mappo.run import MAPPO_Runner
            from algorithms.mappo.types import Experiment

            exp_config = Experiment(**exp_dict)
            runner = MAPPO_Runner(
                exp_config.device,
                batch_dir,
                results_dir,
                trial_id,
                checkpoint,
                exp_config,
                env_config,
            )

        case AlgorithmEnum.MAPPO_VANILLA:
            from algorithms.mappo_vanilla.run import MAPPO_Vanilla_Runner
            from algorithms.mappo_vanilla.types import Experiment

            exp_config = Experiment(**exp_dict)
            runner = MAPPO_Vanilla_Runner(
                exp_config.device,
                batch_dir,
                results_dir,
                trial_id,
                checkpoint,
                exp_config,
                env_config,
            )

        case AlgorithmEnum.MAPPO_JAX:
            from algorithms.mappo_jax.run import MAPPO_JAX_Runner
            from algorithms.mappo_jax.types import Experiment

            if debug:
                jax.config.update("jax_disable_jit", True)

            exp_config = Experiment(**exp_dict)
            runner = MAPPO_JAX_Runner(
                exp_config.device,
                batch_dir,
                results_dir,
                trial_id,
                checkpoint,
                exp_config,
                env_config,
            )

        case AlgorithmEnum.RMAPPO_JAX:
            # The reference MAPPO (marlbenchmark/on-policy) over the same
            # functional JAX envs: value normalization, Huber + clipped value
            # loss, batch-wide advantage normalization, persistent rollouts and
            # a GRU policy/critic trained on chunks (model=rmappo / mappo_mlp).
            from algorithms.rmappo_jax.run import RMAPPO_JAX_Runner
            from algorithms.rmappo_jax.types import Experiment

            if debug:
                jax.config.update("jax_disable_jit", True)

            exp_config = Experiment(**exp_dict)
            runner = RMAPPO_JAX_Runner(
                exp_config.device,
                batch_dir,
                results_dir,
                trial_id,
                checkpoint,
                exp_config,
                env_config,
            )

    if view:
        runner.view()
    elif evaluate:
        runner.evaluate()
    else:
        runner.train()
