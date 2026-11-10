#!/bin/bash
#SBATCH --job-name=irl-comparison
#SBATCH --account=gts-awu36
#SBATCH --qos=embers
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=08:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

# Account/QoS follow this repository's existing Phoenix scripts. Override them
# with sbatch -A/--qos if your allocation differs. Current trainers use CPU.
# Submit from repo root after creating logs: sbatch scripts/irl_comparison.sbatch smoke
# Modes: smoke (short end-to-end check), pilot (full configured comparison).
set -euo pipefail

cd "${SLURM_SUBMIT_DIR:?Submit with sbatch from the repository root}"
: "${SLURM_JOB_ID:?Use sbatch so training runs in a compute allocation}"
mode=${1:-smoke}
if [[ $# -gt 1 || ( "$mode" != smoke && "$mode" != pilot ) ]]; then
    echo 'Usage: sbatch scripts/irl_comparison.sbatch smoke|pilot' >&2
    exit 2
fi
bench_python="$PWD/.venv-airl/bin/python"
if [[ ! -x "$bench_python" ]]; then
    echo 'Missing .venv-airl/bin/python. Run module load anaconda3, then bash scripts/setup_irl_bench_phoenix.sh.' >&2
    exit 2
fi
if [[ ! -f "configs/irl_comparison/$mode.json" ]]; then
    echo "Missing configs/irl_comparison/$mode.json. Sync the benchmark configurations to Phoenix first." >&2
    exit 2
fi
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MPLCONFIGDIR="${TMPDIR:-/tmp}/active-sensing-mpl-${SLURM_JOB_ID}"
export PYTHONUNBUFFERED=1

mkdir -p artifacts/irl_comparison
bench_run_dir=$(mktemp -d "$PWD/artifacts/irl_comparison/${mode}-${SLURM_JOB_ID}.XXXXXX")
echo "Results directory: $bench_run_dir"
echo "Mode: $mode; hostname: $(hostname); Python: $bench_python"
"$bench_python" -I -c 'import imitation, active_sensing_bench; from imitation.algorithms.adversarial.airl import AIRL; print("External imitation:", imitation.__file__); print("Benchmark:", active_sensing_bench.__file__)'

# Checkpoints are generated here; artifacts and virtual environments are not
# version-controlled and cannot be assumed to exist on a fresh cluster checkout.
# Smoke experts are deliberately undertrained and only exercise the pipeline.
expert_args=(--timesteps 200704 --eval-episodes 200)
if [[ "$mode" == smoke ]]; then
    expert_args=(--timesteps 128 --envs 2 --steps-per-env 32 --batch-size 32 --epochs 1 --eval-episodes 2)
fi
for representation in full belief; do
    "$bench_python" -I -m active_sensing_bench.retrain_policy \
        --reward true --representation "$representation" --view-size 3 --seed 0 \
        "${expert_args[@]}" --output "$bench_run_dir/expert_$representation"
done

"$bench_python" -I -m active_sensing_bench.compare \
    --config "configs/irl_comparison/$mode.json" \
    --expert-full "$bench_run_dir/expert_full" \
    --expert-belief "$bench_run_dir/expert_belief" \
    --output "$bench_run_dir/comparison"

"$bench_python" -I -m active_sensing_bench.train_mce \
    --seed 0 --output "$bench_run_dir/mce"

echo "Completed. Comparison: $bench_run_dir/comparison/comparison.csv"
echo "MCE results: $bench_run_dir/mce/evaluation.json"
