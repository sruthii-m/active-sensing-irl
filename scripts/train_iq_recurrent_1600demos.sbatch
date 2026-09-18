#!/bin/bash
#SBATCH --job-name=iq-recurrent-1600demos
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

module load anaconda3 cuda        # check exact module names with `module avail`
mkdir -p logs

source activate active-sensing-irl  #source /path/to/venv/bin/activate

cd "$SLURM_SUBMIT_DIR"

# Data-scaling test, third point: 1600 demo episodes (200 -> 400 -> 800 -> 1600).
# Trend so far: 60% -> 70% -> 79%, near-linear per doubling, no saturation yet.
# This checks whether that trend continues toward the full-state 88% ceiling
# or starts to bend -- the deciding evidence for how much of the belief-MDP
# stage is actually necessary vs. closeable by data alone.
python -u -m imitation.train_iq_recurrent \
    --env partial_obs \
    --demo-path artifacts/demos/forage_partial_obs_demos_1600.pkl \
    --max-steps 200000 \
    --eval-freq 3000 \
    --patience 5 \
    --min-delta 0.04 \
    --plateau-eval-episodes 100 \
    --grid-shape 7 7 3 \
    --chi2-alpha 0.5 \
    --latent-dim 128 \
    --rnn-hidden-dim 256 \
    --save-path artifacts/iq_learn/forage_partial_obs_qnet_1600demos.pt
