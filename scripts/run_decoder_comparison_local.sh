#!/usr/bin/env bash
set -euo pipefail
if [[ "${1:-}" == --help ]]; then
    echo 'Usage: bash scripts/run_decoder_comparison_local.sh [POLICY_SEED=0] [OUTPUT_DIR]'
    echo 'Environment: PYTHON, ROLLOUTS, EPOCHS=300, TEST_EPISODES=200, TEST_SEED=700000'
    exit 0
fi
(( $# <= 2 )) || { echo 'Too many arguments' >&2; exit 2; }
policy_seed=${1:-0}
[[ "$policy_seed" =~ ^[0-9]+$ ]] || { echo 'Policy seed must be nonnegative' >&2; exit 2; }
cd "$(dirname "${BASH_SOURCE[0]}")/.."
rollouts=${ROLLOUTS:-artifacts/belief_decoding/fov3_seed${policy_seed}_probe0/rollouts.npz}
out=${2:-artifacts/decoder_comparison/fov3_seed${policy_seed}}
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export MPLCONFIGDIR="${TMPDIR:-/tmp}/active-sensing-mpl"
"${PYTHON:-python}" -u -m analysis.compare_belief_decoders \
    --rollouts "$rollouts" --output-dir "$out" --epochs "${EPOCHS:-300}" \
    --test-episodes "${TEST_EPISODES:-200}" --test-seed "${TEST_SEED:-700000}"
