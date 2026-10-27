#!/usr/bin/env bash
# Frozen-policy evaluation only; no expert or IQ retraining.
set -euo pipefail
if [[ "${1:-}" == --help ]]; then
    echo 'Usage: bash scripts/run_belief_decoding_local.sh [FOV=3] [POLICY_SEED=0] [OUTPUT_DIR]'
    echo 'Environment: PYTHON, CHECKPOINT, EPISODES=300, EPOCHS=100, PROBE_SEED=0'
    exit 0
fi
(( $# <= 3 )) || { echo 'Too many arguments; use --help.' >&2; exit 2; }
fov=${1:-3}
policy_seed=${2:-0}
case "$fov" in 3|5|7) ;; *) echo 'FOV must be 3, 5 or 7.' >&2; exit 2;; esac
[[ "$policy_seed" =~ ^[0-9]+$ ]] || { echo 'Policy seed must be nonnegative.' >&2; exit 2; }
cd "$(dirname "${BASH_SOURCE[0]}")/.."
out=${3:-artifacts/belief_decoding/fov${fov}_seed${policy_seed}_probe${PROBE_SEED:-0}}
checkpoint=${CHECKPOINT:-artifacts/state_uncertainty/partial_obs_fov${fov}_seed${policy_seed}_local/iq.pt}
[[ -f "$checkpoint" ]] || { echo "Missing checkpoint: $checkpoint" >&2; exit 2; }
[[ ! -e "$out" ]] || { echo "Output already exists: $out. Choose a new directory." >&2; exit 2; }
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export MPLCONFIGDIR="${TMPDIR:-/tmp}/active-sensing-mpl"
"${PYTHON:-python}" -u -m analysis.evaluate_belief_decoding \
    --checkpoint "$checkpoint" --init-temp .001 --grid-shape "$fov" "$fov" 3 \
    --view-size "$fov" --episodes "${EPISODES:-300}" --epochs "${EPOCHS:-100}" \
    --probe-seed "${PROBE_SEED:-0}" --output-dir "$out"
