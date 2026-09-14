#!/bin/bash
#SBATCH --job-name=iq-recurrent
#SBATCH --account=<your-pace-account>      # sacctmgr show associations user=$USER
#SBATCH --qos=inferno                      
#SBATCH -N1
#SBATCH -n1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

module load anaconda3 cuda        # check exact module names with `module avail`
mkdir -p logs

source activate active-sensing-irl  #source /path/to/venv/bin/activate

cd "$SLURM_SUBMIT_DIR"

python -u -m imitation.train_iq_recurrent \
    --env partial_obs \
    --demo-path artifacts/demos/forage_partial_obs_demos.pkl \
    --steps 15000 \
    --grid-shape 7 7 3 \
    --chi2-alpha 5.0 \
    --save-path artifacts/iq_learn/forage_partial_obs_qnet.pt
