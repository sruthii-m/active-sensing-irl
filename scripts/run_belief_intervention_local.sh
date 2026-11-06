#!/usr/bin/env bash
set -euo pipefail
if [[ "${1:-}" == --help ]]; then
    echo 'Usage: bash scripts/run_belief_intervention_local.sh [POLICY_SEED=0] [OUTPUT_ROOT]'
    echo 'Optional: PYTHON, CHECKPOINT, EPISODES=600, PROBE, EXCLUDE_ROLLOUTS, RESET_SEED=400000'
    exit 0
fi
(( $# <= 2 )) || exit 2
seed=${1:-0}
[[ "$seed" =~ ^[0-9]+$ ]] || { echo 'Policy seed must be nonnegative.' >&2; exit 2; }
cd "$(dirname "${BASH_SOURCE[0]}")/.."
root=${2:-artifacts/belief_causal_development/seed${seed}}
[[ ! -e "$root" ]] || { echo "Choose a new output root: $root" >&2; exit 2; }
probe=${PROBE:-artifacts/belief_decoding/robustness_fixed_budget/policy${seed}_probe0/hidden_probe.pt}
[[ -f "$probe" ]] || { echo "Missing frozen decoder: $probe" >&2; exit 2; }
checkpoint=${CHECKPOINT:-artifacts/state_uncertainty/partial_obs_fov3_seed${seed}_local/iq.pt}
exclude=${EXCLUDE_ROLLOUTS:-artifacts/belief_decoding/fov3_seed${seed}_probe0/rollouts.npz}
[[ -f "$checkpoint" && -f "$exclude" ]] || { echo 'Missing checkpoint or excluded probe rollouts.' >&2; exit 2; }
"${PYTHON:-python}" - "$checkpoint" "$probe" "$exclude" <<'PY_CHECK'
import hashlib, json, sys
from pathlib import Path
checkpoint, probe, rollouts = map(Path, sys.argv[1:])
digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
for artifact in (probe, rollouts):
    metadata = json.loads(artifact.with_name('summary.json').read_text())
    if metadata['checkpoint_sha256'] != digest:
        raise SystemExit(f'Policy mismatch: {artifact}. Regenerate probes and rollouts for this checkpoint.')
probe_metadata = json.loads(probe.with_name('summary.json').read_text())
if probe_metadata.get('rollout_sha256') != hashlib.sha256(rollouts.read_bytes()).hexdigest():
    raise SystemExit('EXCLUDE_ROLLOUTS must be the exact rollout dataset used to fit the probe.')
PY_CHECK
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export MPLCONFIGDIR="${TMPDIR:-/tmp}/active-sensing-mpl"
"${PYTHON:-python}" -u -m analysis.evaluate_natural_histories \
    --checkpoint "$checkpoint" \
    --init-temp .001 --grid-shape 3 3 3 --episodes "${EPISODES:-600}" \
    --rollout-seed "${RESET_SEED:-400000}" --exclude-rollouts "$exclude" \
    --output-dir "$root/natural"
"${PYTHON:-python}" -u -m analysis.intervene_belief_memory \
    --natural-run "$root/natural" --probe "$probe" --output-dir "$root/interventions"
