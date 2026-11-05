#!/usr/bin/env bash
# Draw the arena of every rover env group (every conf/env/rover_*.yaml) into
# plotting/rover/arenas/<group>.png, through environments.rover.visualize.
# New groups are picked up automatically: the Python script globs conf/env/.
#
#   bash scripts/visualize_rover_arenas.sh                    # arenas only
#   TRAJECTORIES=1 bash scripts/visualize_rover_arenas.sh     # also episodes of the
#       random and scripted policies, in plotting/rover/trajectories/
#   bash scripts/visualize_rover_arenas.sh --out-dir /tmp/rover --seeds 0 1
#       (any further arguments go to the Python script; see its --help)
#
# The "PJRT_Api already exists" / "Jax plugin configuration error" traceback at
# start-up is the known, harmless CUDA plugin message.
set -euo pipefail

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
cd "$( dirname "$SCRIPT_DIR" )"

args=()
if [[ "${TRAJECTORIES:-0}" != 1 ]]; then
    args+=(--no-trajectories)
fi

exec uv run python -m environments.rover.visualize "${args[@]}" "$@"
