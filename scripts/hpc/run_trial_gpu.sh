#!/bin/bash
BATCH_NAME=$1
EXPERIMENT_NAME=$2
ALGORITHM=$3
ENVIRONMENT=$4
TRIAL_ID=$5

SCRATCH=/nfs/stak/users/agrazvam/hpc-share/tmp
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
EXPERIMENT_SCRIPT="${REPO_DIR}/train.py"

sbatch <<EOT
#!/bin/bash
#SBATCH -J ${TRIAL_ID}_${EXPERIMENT_NAME}_${ALGORITHM}_${BATCH_NAME}
#SBATCH -A kt-lab
#SBATCH --partition=preempt,dgxh
#SBATCH -o ./logs/${BATCH_NAME}_${EXPERIMENT_NAME}_${TRIAL_ID}.out
#SBATCH -e ./logs/${BATCH_NAME}_${EXPERIMENT_NAME}_${TRIAL_ID}.err
#SBATCH -c 2
#SBATCH --cpu-freq=high
#SBATCH --mem=8G
#SBATCH --time=1:00:00
#SBATCH --nodelist=dgxh-[1-4],cn-w-[1-2],cn-t-1,cn-r-[1-6],cn-s-[1-5],cn-gpu[5-7],cn-gpu[10-12],optimus,sail-gpu0,dgx2-[1-2],dgx2-[4-5],cn-x-[1-2],cn-m-[1-2]
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --requeue

export TMPDIR="$SCRATCH"

# Job info, Python module, and a check that a GPU is visible.
source "${REPO_DIR}/scripts/hpc/gpu_job_preamble.sh"

# Prepare the shared environment once with uv sync before submitting jobs.
# Each trial must use that project without installing packages while other
# jobs may already be importing them from the shared filesystem.
cd "${REPO_DIR}" || exit 1
if [[ ! -x "${REPO_DIR}/.venv/bin/python3" ]]; then
    echo "ERROR: Project environment missing; run uv sync --locked in ${REPO_DIR} before submitting jobs."
    exit 1
fi

echo "Using prepared uv runtime in ${REPO_DIR}/.venv"

uv run --no-sync --project "${REPO_DIR}" "${REPO_DIR}/.venv/bin/python3" - <<'PY'
import sys
import torch

print(f"Python executable: {sys.executable}", flush=True)
print(f"PyTorch module file: {getattr(torch, '__file__', None)}", flush=True)
print(f"PyTorch module path: {getattr(torch, '__path__', None)}", flush=True)
if not hasattr(torch, "__version__"):
    raise SystemExit(
        "ERROR: Imported torch has no __version__. Check the module paths above "
        "for shadowing or an incomplete installation. If the project installation "
        "is incomplete, run uv sync --locked --reinstall-package torch with jobs stopped."
    )
print(f"PyTorch version: {torch.__version__}")
print(f"PyTorch CUDA build: {torch.version.cuda}")
available = torch.cuda.is_available()
print(f"PyTorch CUDA available: {available}")
print(f"CUDA device count: {torch.cuda.device_count()}")
print(f"CUDA device name: {torch.cuda.get_device_name(0) if available else 'None'}")
raise SystemExit(0 if available else 1)
PY
if [[ \$? -ne 0 ]]; then
    echo "ERROR: PyTorch runtime check failed; see diagnostics above."
    exit 1
fi

uv run --no-sync --project "${REPO_DIR}" "${REPO_DIR}/.venv/bin/python3" "$EXPERIMENT_SCRIPT" \
    env="$BATCH_NAME" \
    model="$EXPERIMENT_NAME" \
    algorithm="$ALGORITHM" \
    trial_id="$TRIAL_ID" \
    checkpoint=true

EOT
