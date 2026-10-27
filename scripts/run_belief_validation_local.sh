#!/usr/bin/env bash
set -euo pipefail
if [[ "${1:-}" == --help ]]; then
    echo 'Usage: bash scripts/run_belief_validation_local.sh [POLICY_SEED=0] [OUTPUT_DIR]'
    echo 'Environment: PYTHON, ROLLOUTS, EPOCHS=200, PARAMETER_BUDGET=20000'
    exit 0
fi
(( $# <= 2 )) || { echo 'Too many arguments' >&2; exit 2; }
policy_seed=${1:-0}
[[ "$policy_seed" =~ ^[0-9]+$ ]] || { echo 'Policy seed must be nonnegative' >&2; exit 2; }
cd "$(dirname "${BASH_SOURCE[0]}")/.."
rollouts=${ROLLOUTS:-artifacts/belief_decoding/fov3_seed${policy_seed}_probe0/rollouts.npz}
out=${2:-artifacts/belief_validation/fov3_seed${policy_seed}}
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export MPLCONFIGDIR="${TMPDIR:-/tmp}/active-sensing-mpl"
"${PYTHON:-python}" -u -m analysis.validate_belief_consistency \
    --rollouts "$rollouts" --output-dir "$out" --epochs "${EPOCHS:-200}" \
    --parameter-budget "${PARAMETER_BUDGET:-20000}" --seeds 0 1 2
