#!/usr/bin/env bash
# Local CPU pipeline; uses the currently activated Python environment.
set -euo pipefail

usage() {
    echo "Usage: bash scripts/run_state_uncertainty_local.sh [FOV=3] [SEED=0] [OUTPUT_DIR]"
    echo "FOV: 3, 5, or 7. OUTPUT_DIR must not already exist."
    echo "Optional: PYTHON=/path/to/python to select an interpreter."
}
if [[ "${1:-}" == --help || "${1:-}" == -h ]]; then
    usage
    exit 0
fi
if (( $# > 3 )); then usage >&2; exit 2; fi
fov=${1:-3}
seed=${2:-0}
case "$fov" in 3|5|7) ;; *) usage >&2; exit 2;; esac
[[ "$seed" =~ ^(0|[1-9][0-9]*)$ ]] || { echo 'Seed must be a nonnegative integer.' >&2; exit 2; }

cd "$(dirname "${BASH_SOURCE[0]}")/.."
run=${3:-artifacts/state_uncertainty/partial_obs_fov${fov}_seed${seed}_local}
if [[ -e "$run" ]]; then
    echo "Run directory already exists: $run" >&2
    echo 'Use a new seed or supply a different output directory; nothing was overwritten.' >&2
    exit 2
fi
python_bin=${PYTHON:-python}
command -v "$python_bin" >/dev/null || { echo "Interpreter not found: $python_bin. Activate your Python environment." >&2; exit 2; }
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
"$python_bin" -c 'import torch, minigrid, sb3_contrib, matplotlib'
mkdir -p "$(dirname "$run")"
mkdir "$run"
export MPLCONFIGDIR="$PWD/$run/mpl_cache"
if [[ "$run" == /* ]]; then export MPLCONFIGDIR="$run/mpl_cache"; fi

stage=setup
trap 'code=$?; if (( code != 0 )); then printf "failed: %s (exit %s)\n" "$stage" "$code" > "$run/status.txt"; echo "Stopped during $stage. See $run/$stage.log" >&2; fi' EXIT
run_stage() {
    stage=$1
    shift
    printf 'running: %s\n' "$stage" > "$run/status.txt"
    echo "Starting $stage. Log: $run/$stage.log"
    printf '%q ' "$@" >> "$run/commands.sh"
    printf '\n' >> "$run/commands.sh"
    if "$@" > "$run/$stage.log" 2>&1; then
        echo "Finished $stage."
        tail -n 2 "$run/$stage.log"
    else
        code=$?
        tail -n 20 "$run/$stage.log" >&2
        return "$code"
    fi
}

echo "Zero distractors; FOV=$fov; seed=$seed; all seven actions."
echo "Outputs: $run"
common=(--size 10 --num-distractors 0 --view-size "$fov")
run_stage expert_train "$python_bin" -u -m experts.train_partial_obs "${common[@]}" \
    --timesteps 250000 --eval-episodes 100 --seed "$seed" --save-path "$run/expert"
run_stage train_demos "$python_bin" -u -m imitation.generate_partial_obs_demos "${common[@]}" \
    --model-path "$run/expert.zip" --num-episodes 400 --seed 100000 --keep-all --out-path "$run/train.pkl"

stage=heldout_seed
heldout_seed=$("$python_bin" - "$run/train.pkl" <<'PY'
import pickle, sys
with open(sys.argv[1], 'rb') as f:
    data = pickle.load(f)
print(data['seed_start'] + data['attempts'] + 1000)
PY
)
run_stage heldout_demos "$python_bin" -u -m imitation.generate_partial_obs_demos "${common[@]}" \
    --model-path "$run/expert.zip" --num-episodes 100 --seed "$heldout_seed" \
    --keep-all --out-path "$run/heldout.pkl"
stage=validation_seed
validation_seed=$("$python_bin" - "$run/heldout.pkl" <<'PYSEED'
import pickle, sys
with open(sys.argv[1], 'rb') as f:
    data = pickle.load(f)
print(data['seed_start'] + data['attempts'] + 1000)
PYSEED
)
run_stage validation_demos "$python_bin" -u -m imitation.generate_partial_obs_demos "${common[@]}" \
    --model-path "$run/expert.zip" --num-episodes 100 --seed "$validation_seed" \
    --keep-all --out-path "$run/validation.pkl"
run_stage iq_train "$python_bin" -u -m imitation.train_iq_recurrent "${common[@]}" \
    --env partial_obs --demo-path "$run/train.pkl" --validation-demo-path "$run/validation.pkl" --checkpoint-dir "$run/checkpoints" --seed "$seed" \
    --grid-shape "$fov" "$fov" 3 --latent-dim 128 --rnn-hidden-dim 256 \
    --episodes-per-batch 32 --max-steps 200000 --eval-freq 3000 \
    --patience 5 --min-delta 0.001 --eval-episodes 100 \
    --chi2-alpha 0.5 --init-temp 0.001 --gamma 0.99 --device cpu --save-path "$run/iq.pt"

policy=(--checkpoint "$run/iq.pt" --init-temp 0.001 --gamma 0.99 --grid-shape "$fov" "$fov" 3)
run_stage likelihood "$python_bin" -u -m analysis.expert_likelihood "${policy[@]}" \
    --expert-model "$run/expert.zip" --heldout-demos "$run/heldout.pkl" --output "$run/likelihood.json"
run_stage alignment "$python_bin" -u -m analysis.evaluate_belief_alignment "${policy[@]}" \
    --size 10 --view-size "$fov" --episodes 100 --seed 10000 --output "$run/alignment.csv"
run_stage reward_map "$python_bin" -u -m analysis.reward_maps "${policy[@]}" \
    --env partial_obs --size 10 --view-size "$fov" --seed 10000 --output "$run/reward"

run_stage learning_progress "$python_bin" -u -m analysis.evaluate_learning_progress \
    --manifest "$run/checkpoints/manifest.json" --expert-model "$run/expert.zip" \
    --episodes 20 --seed 10000 --output-dir "$run/learning_progress"

printf 'complete\n'  > "$run/status.txt"
echo "Complete. Reports: $run/likelihood.json and $run/alignment.json"
