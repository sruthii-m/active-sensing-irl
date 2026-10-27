#!/usr/bin/env bash
set -euo pipefail
if [[ "${1:-}" == --help ]]; then
    echo 'Usage: bash scripts/run_selected_rank_local.sh [POLICY_SEED=0] [OUTPUT_ROOT]'
    echo 'Environment: PYTHON, ROLLOUTS, CHECKPOINT, EPISODES=300, RESET_SEED=600000'
    exit 0
fi
(( $# <= 2 )) || { echo 'Too many arguments' >&2; exit 2; }
policy_seed=${1:-0}
[[ "$policy_seed" =~ ^[0-9]+$ ]] || { echo 'Policy seed must be nonnegative' >&2; exit 2; }
cd "$(dirname "${BASH_SOURCE[0]}")/.."
rollouts=${ROLLOUTS:-artifacts/belief_decoding/fov3_seed${policy_seed}_probe0/rollouts.npz}
checkpoint=${CHECKPOINT:-artifacts/state_uncertainty/partial_obs_fov3_seed${policy_seed}_local/iq.pt}
out=${2:-artifacts/selected_rank/fov3_seed${policy_seed}}
[[ ! -e "$out" ]] || { echo 'Choose a new output root.' >&2; exit 2; }
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export MPLCONFIGDIR="${TMPDIR:-/tmp}/active-sensing-mpl"
"${PYTHON:-python}" -u -m analysis.select_belief_rank --rollouts "$rollouts" --output-dir "$out/selection"
if "${PYTHON:-python}" -c 'import json,sys; sys.exit(0 if json.load(open(sys.argv[1]))["selected_rank"] == 0 else 1)' "$out/selection/selection.json"; then
    echo 'Validation selected rank zero. No fresh collection or intervention is warranted.'
    exit 0
fi
"${PYTHON:-python}" -u -m analysis.evaluate_matched_histories \
    --checkpoint "$checkpoint" --init-temp .001 --grid-shape 3 3 3 --view-size 3 \
    --episodes "${EPISODES:-300}" --seed "${RESET_SEED:-600000}" --output-dir "$out/histories"
"${PYTHON:-python}" -u -m analysis.validate_matched_histories \
    --histories "$out/histories" --output-dir "$out/comparison"
"${PYTHON:-python}" -u -m analysis.transfer_belief_subspace \
    --comparison "$out/comparison" --rollouts "$rollouts" \
    --selection "$out/selection/selection.json" --output-dir "$out/transfer"
