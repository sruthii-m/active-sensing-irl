#!/bin/bash
#SBATCH --job-name=irl-comparison-1m
#SBATCH --account=gts-awu36
#SBATCH --qos=embers
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=08:00:00
#SBATCH --no-requeue
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

# Submit from the repository root, with logs/ already present.
# First run: sbatch scripts/irl_comparison_1m.sbatch EXPERT_ROOT
# Resume:    sbatch scripts/irl_comparison_1m.sbatch EXPERT_ROOT COMPARISON_DIR
# EXPERT_ROOT contains full_seed0/ and belief_seed0/ from the validated controls.
# No expert training happens here. Completed stages survive job interruption.
set -euo pipefail

if [[ $# -lt 1 || $# -gt 2 ]]; then
    echo 'Usage: sbatch scripts/irl_comparison_1m.sbatch EXPERT_ROOT [COMPARISON_DIR_TO_RESUME]' >&2
    exit 2
fi
cd "${SLURM_SUBMIT_DIR:?Submit with sbatch from the repository root}"
: "${SLURM_JOB_ID:?Use sbatch so training runs in a compute allocation}"
bench_python="$PWD/.venv-airl/bin/python"
bench_config="$PWD/configs/irl_comparison/pilot_1m.json"
bench_runner="$PWD/scripts/run_irl_comparison.py"
if [[ ! -x "$bench_python" ]]; then
    echo 'Missing .venv-airl/bin/python. Run scripts/setup_irl_bench_phoenix.sh first.' >&2
    exit 2
fi
for required in "$bench_config" "$bench_runner"; do
    if [[ ! -f "$required" ]]; then
        echo "Missing $required. Sync the new configuration and launch scripts to Phoenix." >&2
        exit 2
    fi
done
for representation in full belief; do
    for artifact in policy.zip config.json evaluation.json; do
        if [[ ! -f "$1/${representation}_seed0/$artifact" ]]; then
            echo "Missing expert artifact: $1/${representation}_seed0/$artifact" >&2
            exit 2
        fi
    done
done

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MPLCONFIGDIR="${TMPDIR:-/tmp}/active-sensing-mpl-${SLURM_JOB_ID}"
bench_resume=()
if [[ $# -eq 2 ]]; then
    bench_output=$2
    bench_resume=(--resume)
else
    mkdir -p artifacts/irl_comparison
    bench_run_dir=$(mktemp -d "$PWD/artifacts/irl_comparison/pilot-1m-${SLURM_JOB_ID}.XXXXXX")
    bench_output="$bench_run_dir/comparison"
fi
echo "Comparison directory: $bench_output"
echo "Expert root: $1"
echo 'Recovery/control budget: 1003520; AIRL rounds: 49; control-success gate: 0.8'
echo 'On interruption, resubmit with the same expert root and comparison directory as the second argument.'

# Execute this source script explicitly. -I keeps imports on the installed
# benchmark/external imitation packages; no wheel reinstall is needed here.
"$bench_python" -I -u "$bench_runner" \
    --config "$bench_config" --experts "$1" --output "$bench_output" "${bench_resume[@]}"
