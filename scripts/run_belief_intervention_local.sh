#!/usr/bin/env bash
set -euo pipefail
if [[ "${1:-}" == --help ]]; then
    echo 'Usage: bash scripts/run_belief_intervention_local.sh [POLICY_SEED=0] [OUTPUT_ROOT]'
    echo 'Optional: PYTHON, EPISODES=600, PROBE=/path/to/hidden_probe.pt'
    exit 0
fi
(( $# <= 2 )) || exit 2
seed=${1:-0}
case "$seed" in 0|1) ;; *) echo 'This wrapper supports the existing FOV 3 policy seeds 0 and 1.' >&2; exit 2;; esac
cd "$(dirname "${BASH_SOURCE[0]}")/.."
root=${2:-artifacts/belief_causal_development/seed${seed}}
[[ ! -e "$root" ]] || { echo "Choose a new output root: $root" >&2; exit 2; }
probe=${PROBE:-artifacts/belief_decoding/robustness_fixed_budget/policy${seed}_probe0/hidden_probe.pt}
[[ -f "$probe" ]] || { echo "Missing frozen decoder: $probe" >&2; exit 2; }
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export MPLCONFIGDIR="${TMPDIR:-/tmp}/active-sensing-mpl"
"${PYTHON:-python}" -u -m analysis.evaluate_natural_histories \
    --checkpoint "artifacts/state_uncertainty/partial_obs_fov3_seed${seed}_local/iq.pt" \
    --init-temp .001 --grid-shape 3 3 3 --episodes "${EPISODES:-600}" \
    --exclude-rollouts "artifacts/belief_decoding/fov3_seed${seed}_probe0/rollouts.npz" \
    --output-dir "$root/natural"
"${PYTHON:-python}" -u -m analysis.intervene_belief_memory \
    --natural-run "$root/natural" --probe "$probe" --output-dir "$root/interventions"
