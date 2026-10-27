#!/usr/bin/env bash
set -euo pipefail
if [[ "${1:-}" == --help ]]; then
    echo 'Usage: bash scripts/run_subspace_recovery_local.sh [POLICY_DATASET=0] [OUTPUT_DIR]'
    echo 'Environment: PYTHON, ROLLOUTS'
    exit 0
fi
(( $# <= 2 )) || { echo 'Too many arguments' >&2; exit 2; }
policy_seed=${1:-0}
[[ "$policy_seed" =~ ^[0-9]+$ ]] || { echo 'Dataset index must be nonnegative' >&2; exit 2; }
cd "$(dirname "${BASH_SOURCE[0]}")/.."
rollouts=${ROLLOUTS:-artifacts/belief_decoding/fov3_seed${policy_seed}_probe0/rollouts.npz}
out=${2:-artifacts/subspace_recovery/fov3_seed${policy_seed}}
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export MPLCONFIGDIR="${TMPDIR:-/tmp}/active-sensing-mpl"
"${PYTHON:-python}" -u -m analysis.validate_subspace_recovery \
    --rollouts "$rollouts" --output-dir "$out" --seeds 0 1 2
