#!/usr/bin/env bash
set -euo pipefail
if [[ "${1:-}" == --help ]]; then
    echo 'Usage: bash scripts/run_belief_consistency_local.sh [FOV=3] [POLICY_SEED=0] [OUTPUT_DIR]'
    echo 'Environment: PYTHON, ROLLOUTS, EPOCHS=100, PROBE_SEED=0, RANDOM_SEED=0'
    exit 0
fi
(( $# <= 3 )) || { echo 'Too many arguments' >&2; exit 2; }
fov=${1:-3}
policy_seed=${2:-0}
[[ "$fov" =~ ^(3|5|7)$ && "$policy_seed" =~ ^[0-9]+$ ]] || { echo 'Invalid FOV or policy seed' >&2; exit 2; }
cd "$(dirname "${BASH_SOURCE[0]}")/.."
rollouts=${ROLLOUTS:-artifacts/belief_decoding/fov${fov}_seed${policy_seed}_probe0/rollouts.npz}
out=${3:-artifacts/belief_consistency/fov${fov}_seed${policy_seed}_probe${PROBE_SEED:-0}_random${RANDOM_SEED:-0}}
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export MPLCONFIGDIR="${TMPDIR:-/tmp}/active-sensing-mpl"
"${PYTHON:-python}" -u -m analysis.evaluate_belief_consistency \
    --rollouts "$rollouts" --output-dir "$out" --epochs "${EPOCHS:-100}" \
    --probe-seed "${PROBE_SEED:-0}" --random-seed "${RANDOM_SEED:-0}"
