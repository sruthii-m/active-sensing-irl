#!/bin/bash
#SBATCH --job-name=state-uncertainty
#SBATCH --account=gts-awu36
#SBATCH --qos=embers
#SBATCH -N1
#SBATCH -n1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

# Submit from repo root: sbatch scripts/state_uncertainty.sbatch expert partial_obs 7 0
# Arguments: stage {expert,demos,iq,evaluate}, mode {partial_obs,full_state}, FOV, seed.
# Resource/module settings follow this repository's existing PACE scripts.
set -eo pipefail
module load anaconda3 cuda
source activate active-sensing-irl
set -u
cd "${SLURM_SUBMIT_DIR:?Submit from the repository root}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export MPLCONFIGDIR="${TMPDIR:-/tmp}/active-sensing-mpl-${SLURM_JOB_ID}"

stage=${1:?Specify expert, demos, iq, or evaluate}
mode=${2:-partial_obs}
fov=${3:-7}
seed=${4:-0}
case "$mode" in partial_obs|full_state) ;; *) echo 'Invalid mode' >&2; exit 2;; esac
case "$fov" in 3|5|7) ;; *) echo 'FOV must be 3, 5, or 7' >&2; exit 2;; esac
[[ "$seed" =~ ^[0-9]+$ ]] || { echo 'Seed must be nonnegative' >&2; exit 2; }
run="artifacts/state_uncertainty/${mode}_fov${fov}_seed${seed}"
mkdir -p "$run"
common=(--size 10 --num-distractors 0 --view-size "$fov")
cnn=()
if [[ "$mode" == partial_obs ]]; then cnn=(--grid-shape "$fov" "$fov" 3); fi
# Record command context alongside each stage's Slurm log.
printf '%s\n' "stage=$stage mode=$mode fov=$fov seed=$seed run=$run"
git rev-parse HEAD > "$run/${stage}_commit.txt"

case "$stage" in
  expert)
    python -u -m "experts.train_${mode}" "${common[@]}" \
      --timesteps 250000 --eval-episodes 100 --seed "$seed" \
      --save-path "$run/expert"
    ;;
  demos)
    # Generate fresh demos in the same condition. No reuse of old distractor data.
    python -u -m "imitation.generate_${mode}_demos" "${common[@]}" \
      --model-path "$run/expert" --num-episodes 400 --seed 100000 \
      --out-path "$run/train.pkl"
    # Avoid overlap even if expert failures required many reset attempts.
    heldout_seed=$(python - "$run/train.pkl" <<'PY'
import pickle, sys
with open(sys.argv[1], 'rb') as f:
    data = pickle.load(f)
print(data['seed_start'] + data['attempts'] + 1000)
PY
)
    python -u -m "imitation.generate_${mode}_demos" "${common[@]}" \
      --model-path "$run/expert" --num-episodes 100 --seed "$heldout_seed" \
      --out-path "$run/heldout.pkl"
    ;;
  iq)
    if [[ "$mode" == partial_obs ]]; then
      python -u -m imitation.train_iq_recurrent "${common[@]}" \
        --env partial_obs --demo-path "$run/train.pkl" --seed "$seed" \
        --max-steps 200000 --eval-freq 3000 --patience 5 --min-delta 0.04 \
        --plateau-eval-episodes 100 --eval-episodes 100 "${cnn[@]}" \
        --chi2-alpha 0.5 --latent-dim 128 --rnn-hidden-dim 256 \
        --init-temp 0.001 --gamma 0.99 --save-path "$run/iq.pt"
    else
      python -u -m imitation.train_iq_full_state "${common[@]}" \
        --demo-path "$run/train.pkl" --seed "$seed" --steps 20000 \
        --eval-episodes 100 --init-temp 0.001 --gamma 0.99 --save-path "$run/iq.pt"
    fi
    ;;
  evaluate)
    policy=(--checkpoint "$run/iq.pt" --init-temp 0.001 --gamma 0.99 "${cnn[@]}")
    python -u -m analysis.expert_likelihood "${policy[@]}" \
      --heldout-demos "$run/heldout.pkl" --output "$run/likelihood.json"
    # Shared layouts across FOVs; demos start at 100000, evaluation at 10000.
    eval_seed=10000
    if [[ "$mode" == partial_obs ]]; then
      python -u -m analysis.evaluate_belief_alignment "${policy[@]}" \
        --view-size "$fov" --episodes 100 --seed "$eval_seed" \
        --output "$run/alignment.csv"
    fi
    for offset in 0 1 2; do
      python -u -m analysis.reward_maps "${policy[@]}" --env "$mode" \
        --view-size "$fov" --seed "$((eval_seed + offset))" \
        --output "$run/reward_${offset}"
    done
    ;;
  *) echo 'Stage must be expert, demos, iq, or evaluate' >&2; exit 2;;
esac
