#!/bin/bash
#SBATCH --job-name=iq-recurrent
#SBATCH --account=gts-awu36
#SBATCH --qos=inferno
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

python -u -m imitation.train_iq_recurrent \
    --env partial_obs \
    --demo-path artifacts/demos/forage_partial_obs_demos.pkl \
    --max-steps 200000 \
    --eval-freq 3000 \
    --patience 5 \
    --min-delta 0.04 \
    --plateau-eval-episodes 100 \
    --grid-shape 7 7 3 \
    --chi2-alpha 0.5 \
    --latent-dim 128 \
    --rnn-hidden-dim 256 \
    --save-path artifacts/iq_learn/forage_partial_obs_qnet_bigger.pt
