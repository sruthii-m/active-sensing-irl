#!/usr/bin/env bash
set -euo pipefail
if [[ "${1:-}" == --help ]]; then
    echo 'Usage: bash scripts/run_matched_validation_local.sh [POLICY_SEED=0] [OUTPUT_ROOT]'
    echo 'Environment: PYTHON, CHECKPOINT, EPISODES=300, RESET_SEED=500000'
    exit 0
fi
(( $# <= 2 )) || { echo 'Too many arguments' >&2; exit 2; }
policy_seed=${1:-0}
[[ "$policy_seed" =~ ^[0-9]+$ ]] || { echo 'Policy seed must be nonnegative' >&2; exit 2; }
cd "$(dirname "${BASH_SOURCE[0]}")/.."
out=${2:-artifacts/matched_validation/fov3_seed${policy_seed}}
checkpoint=${CHECKPOINT:-artifacts/state_uncertainty/partial_obs_fov3_seed${policy_seed}_local/iq.pt}
[[ ! -e "$out" ]] || { echo "Choose a new output root: $out" >&2; exit 2; }
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export MPLCONFIGDIR="${TMPDIR:-/tmp}/active-sensing-mpl"
"${PYTHON:-python}" -u -m analysis.evaluate_matched_histories \
    --checkpoint "$checkpoint" --init-temp .001 --grid-shape 3 3 3 \
    --view-size 3 --episodes "${EPISODES:-300}" --seed "${RESET_SEED:-500000}" \
    --output-dir "$out/histories"
"${PYTHON:-python}" -u -m analysis.validate_matched_histories \
    --histories "$out/histories" --output-dir "$out/comparison"
