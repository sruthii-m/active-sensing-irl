#!/usr/bin/env bash
set -euo pipefail
if [[ "${1:-}" == --help ]]; then
    echo 'Usage: bash scripts/run_belief_transfer_local.sh [POLICY_SEED=0] [OUTPUT_DIR]'
    echo 'Environment: PYTHON, COMPARISON, ROLLOUTS'
    exit 0
fi
(( $# <= 2 )) || { echo 'Too many arguments' >&2; exit 2; }
policy_seed=${1:-0}
[[ "$policy_seed" =~ ^[0-9]+$ ]] || { echo 'Policy seed must be nonnegative' >&2; exit 2; }
cd "$(dirname "${BASH_SOURCE[0]}")/.."
comparison=${COMPARISON:-artifacts/matched_validation/fov3_seed${policy_seed}/comparison}
rollouts=${ROLLOUTS:-artifacts/belief_decoding/fov3_seed${policy_seed}_probe0/rollouts.npz}
out=${2:-artifacts/belief_transfer/fov3_seed${policy_seed}}
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export MPLCONFIGDIR="${TMPDIR:-/tmp}/active-sensing-mpl"
"${PYTHON:-python}" -u -m analysis.transfer_belief_subspace \
    --comparison "$comparison" --rollouts "$rollouts" --output-dir "$out"
