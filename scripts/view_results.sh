#!/usr/bin/env bash
# Render the trained policy (train.py view=true) of one trial of every
# experiment on disk: every experiments/results/<env group>/<model group>/<TRIAL>.
# Each run writes, inside that trial's directory, logs/rover/arena.png and
# logs/rover/trajectory_seed<k>.png (3 deterministic episodes, seeds = the
# trial's seed + 1..3), replacing what an earlier view left there.
#
#   bash scripts/view_results.sh                       # trial 0 of everything
#   bash scripts/view_results.sh rover_4a_2poi_c2 ...  # only these env groups
#   DRY_RUN=1 bash scripts/view_results.sh             # list the runs only
#   TRIAL=3 N_PARALLEL=8 bash scripts/view_results.sh
#
# Environment variables:
#   TRIAL       trial id to render (default 0).
#   N_PARALLEL  runs at once (default 4). A run takes ~5 s, mostly JAX start-up,
#               and ~1.4 GB of RAM on the CPU.
#   DEVICE      cpu (default, JAX_PLATFORMS=cpu) or gpu. The CPU is as fast for
#               these short rollouts and leaves the GPU to training.
#   ALGORITHM   mappo_jax or rmappo_jax for every run. By default it comes from
#               the model group, since the results path does not record it:
#               groups that set use_recurrent_policy are rmappo_jax's, the rest
#               mappo_jax's. A wrong guess fails loudly (each stack rejects the
#               other's model groups with a TypeError).
#   DRY_RUN=1   print each run (algorithm, config warnings) without running it.
#
# Output: one line per run in the terminal, then a summary. Each run's full
# output is in logs/view_results/<timestamp>/<env>__<model>__<trial>.log, and
# summary.tsv there has one row per run (status, run, detail).
#
# ⚠ view builds the env from the current conf/ yamls, not the ones the policy
# trained under (the results do not store them). A run whose env yaml, a yaml
# that env inherits, or model yaml is newer than its saved model is flagged
# "config newer than model": its figures may show the policy in an env it was
# never trained in.
set -uo pipefail

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
REPO_DIR=$( dirname "$SCRIPT_DIR" )
cd "$REPO_DIR"

TRIAL=${TRIAL:-0}
N_PARALLEL=${N_PARALLEL:-4}
DEVICE=${DEVICE:-cpu}
ALGORITHM=${ALGORITHM:-}
DRY_RUN=${DRY_RUN:-0}
RESULTS_DIR="${REPO_DIR}/experiments/results"
CONF_DIR="${REPO_DIR}/conf"

if ! [[ $N_PARALLEL =~ ^[1-9][0-9]*$ ]]; then
    echo "error: N_PARALLEL must be a positive integer, got '${N_PARALLEL}'" >&2
    exit 1
fi
case $DEVICE in
    cpu) export JAX_PLATFORMS=cpu ;;
    gpu)
        unset JAX_PLATFORMS
        # JAX otherwise reserves 75% of the GPU per process (see train_batches.sh).
        if (( N_PARALLEL > 1 )) && [[ -z "${XLA_PYTHON_CLIENT_MEM_FRACTION:-}${XLA_PYTHON_CLIENT_PREALLOCATE:-}${XLA_PYTHON_CLIENT_ALLOCATOR:-}" ]]; then
            export XLA_PYTHON_CLIENT_PREALLOCATE=false
        fi
        ;;
    *) echo "error: DEVICE must be cpu or gpu, got '${DEVICE}'" >&2; exit 1 ;;
esac

source "${SCRIPT_DIR}/check_conf_groups.sh"

LOG_DIR="${REPO_DIR}/logs/view_results/$(date +%Y%m%d_%H%M%S)"
mkdir -p "$LOG_DIR"
: > "${LOG_DIR}/summary.tsv"

# record <status> <run> <detail>: one summary.tsv row and one terminal line.
record() {
    printf '%s\t%s\t%s\n' "$1" "$2" "$3" >> "${LOG_DIR}/summary.tsv"
    printf '%-8s %s  %s\n' "$1" "$2" "$3"
}

# The yaml of env group $1 and, recursively, every env group in its defaults
# list (the variants inherit their parent group that way).
env_config_chain() {
    local file="${CONF_DIR}/env/$1.yaml" parent
    [[ -f $file ]] || return 0
    echo "$file"
    for parent in $(awk '/^defaults:/ {d = 1; next}
                         d && /^[^ \t#]/ {exit}
                         d && $1 == "-" && $2 != "_self_" {print $2}' "$file"); do
        env_config_chain "$parent"
    done
}

algorithm_for() {
    if [[ -n $ALGORITHM ]]; then
        echo "$ALGORITHM"
    elif grep -q 'use_recurrent_policy' "${CONF_DIR}/model/$1.yaml"; then
        echo rmappo_jax
    else
        echo mappo_jax
    fi
}

# view_one <env group> <model group>
view_one() {
    local env_group=$1 model_group=$2
    local run="${env_group}/${model_group}/${TRIAL}"
    local models_dir="${RESULTS_DIR}/${run}/models"
    local log="${LOG_DIR}/${env_group}__${model_group}__${TRIAL}.log"
    local model_file="${models_dir}/models_finished.msgpack" notes=""

    if [[ ! -f $model_file ]]; then
        model_file="${models_dir}/models_checkpoint.msgpack"
        notes+="; unfinished, using the checkpoint"
    fi
    if [[ ! -f $model_file ]]; then
        record skipped "$run" "no saved model"
        return 0
    fi
    if ! check_conf_groups "$CONF_DIR" "env/${env_group}" "model/${model_group}" 2> "$log"; then
        record skipped "$run" "config group missing (see ${log#"${REPO_DIR}"/})"
        return 0
    fi

    local algorithm f newer=()
    algorithm=$(algorithm_for "$model_group")
    for f in $(env_config_chain "$env_group") "${CONF_DIR}/model/${model_group}.yaml"; do
        [[ $f -nt $model_file ]] && newer+=("${f#"${CONF_DIR}"/}")
    done
    if (( ${#newer[@]} > 0 )); then
        notes+="; ⚠ config newer than model: ${newer[*]}"
    fi

    if [[ $DRY_RUN == 1 ]]; then
        record planned "$run" "algorithm=${algorithm}${notes}"
        return 0
    fi

    if uv run python train.py \
            env="$env_group" model="$model_group" algorithm="$algorithm" \
            trial_id="$TRIAL" view=true > "$log" 2>&1; then
        local returns
        returns=$(sed -n 's/.*(return \([^)]*\)).*/\1/p' "$log" | paste -sd' ' -)
        record ok "$run" "returns ${returns}${notes}"
    else
        record failed "$run" "exit $?, see ${log#"${REPO_DIR}"/}${notes}"
    fi
}

export TRIAL ALGORITHM DRY_RUN RESULTS_DIR CONF_DIR REPO_DIR LOG_DIR
export -f record env_config_chain algorithm_for view_one check_conf_groups

# Every <env group>/<model group> with this trial on disk, optionally only the
# env groups named on the command line.
runs=()
for trial_dir in "${RESULTS_DIR}"/*/*/"${TRIAL}"; do
    [[ -d $trial_dir ]] || continue
    model_group=$(basename "$(dirname "$trial_dir")")
    env_group=$(basename "$(dirname "$(dirname "$trial_dir")")")
    if (( $# > 0 )) && [[ " $* " != *" ${env_group} "* ]]; then
        continue
    fi
    runs+=("$env_group" "$model_group")
done
for wanted in "$@"; do
    [[ -d "${RESULTS_DIR}/${wanted}" ]] || echo "warning: no results for env group '${wanted}'" >&2
done
if (( ${#runs[@]} == 0 )); then
    echo "error: no results with trial ${TRIAL} under ${RESULTS_DIR}${*:+ for: $*}" >&2
    exit 1
fi

echo "$(( ${#runs[@]} / 2 )) runs of trial ${TRIAL}, ${N_PARALLEL} at once on ${DEVICE}; logs in ${LOG_DIR#"${REPO_DIR}"/}"
printf '%s\n' "${runs[@]}" \
    | xargs -d '\n' -n 2 -P "$N_PARALLEL" bash -c 'view_one "$1" "$2"' _

summary="${LOG_DIR}/summary.tsv"
count() { awk -F'\t' -v s="$1" '$1 == s' "$summary" | wc -l; }
n_failed=$(count failed)
n_stale=$(grep -c 'config newer than model' "$summary")
echo
echo "summary: $(count ok) ok, ${n_failed} failed, $(count skipped) skipped$( [[ $DRY_RUN == 1 ]] && echo ", $(count planned) planned (dry run)")"
if (( n_stale > 0 )); then
    echo "${n_stale} runs use a config edited after their model was saved:"
    awk -F'\t' '/config newer than model/ {print "  " $2}' "$summary" | sort
fi
if (( n_failed > 0 )); then
    echo "failed:"
    awk -F'\t' '$1 == "failed" {print "  " $2 "  " $3}' "$summary" | sort
    exit 1
fi
[[ $DRY_RUN == 1 ]] || echo "figures: experiments/results/<env>/<model>/${TRIAL}/logs/rover/"
