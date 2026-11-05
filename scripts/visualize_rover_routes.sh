#!/usr/bin/env bash
# Draw the planned scripted team's episodes (ROUTE_PLANS in
# environments/rover/policies.py) for every rover env group that has a plan,
# into plotting/rover/trajectories/<group>_routed_seed<k>.png, through
# environments.rover.scripted_steps. It also prints each plan's statistics
# (return, steps to solve) over the seeds it runs.
#
#   bash scripts/visualize_rover_routes.sh                     # seeds 0-2 of every group
#   bash scripts/visualize_rover_routes.sh --figures 5         # seeds 0-4
#   bash scripts/visualize_rover_routes.sh --variants rover_6a_6poi_c3 rover_4a_4poi_c1234
#   bash scripts/visualize_rover_routes.sh --n-seeds 1000      # statistics over 1000 seeds
#       (any arguments go to the Python script and override the defaults
#       below; see its --help)
#
# The "PJRT_Api already exists" / "Jax plugin configuration error" traceback at
# start-up is the known, harmless CUDA plugin message.
set -euo pipefail

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
cd "$( dirname "$SCRIPT_DIR" )"

# Statistics over the drawn seeds only, so the run stays quick; the last
# occurrence of a flag wins, so "$@" overrides these.
exec uv run python -m environments.rover.scripted_steps --figures 3 --n-seeds 3 "$@"
