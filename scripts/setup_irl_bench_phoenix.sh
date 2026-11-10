#!/usr/bin/env bash
# Run from a Phoenix checkout after: module load anaconda3
# Creates a separate Python 3.11 prefix; no uv or copied Mac environment needed.
set -euo pipefail

cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."
bench_root=$PWD
bench_prefix="$bench_root/.venv-airl"
bench_python="$bench_prefix/bin/python"

if [[ $(uname -s) != Linux ]]; then
    echo 'This setup script is for Linux/Phoenix. Use the local installation instructions on macOS.' >&2
    exit 2
fi
for required in pyproject.toml airl_runner/requirements.txt active_sensing_bench/compare.py; do
    if [[ ! -f "$required" ]]; then
        echo "Missing $required. Sync the benchmark source files to Phoenix first." >&2
        exit 2
    fi
done

mkdir -p artifacts/irl_comparison logs
if [[ ! -x "$bench_python" ]]; then
    if [[ -e "$bench_prefix" ]]; then
        echo "$bench_prefix exists without a working Python. Inspect it before recreating the environment." >&2
        exit 2
    fi
    if ! command -v conda >/dev/null 2>&1; then
        echo 'Conda is unavailable. Run: module load anaconda3, then rerun this script.' >&2
        exit 2
    fi
    conda create --yes --prefix "$bench_prefix" --override-channels --channel conda-forge python=3.11 pip
fi

"$bench_python" -I -c 'import sys; assert sys.version_info[:2] == (3, 11), "The benchmark requires its own Python 3.11 environment"'
"$bench_python" -I -m pip install --upgrade pip
# The current benchmark explicitly runs on CPU. Avoid downloading unused CUDA packages.
"$bench_python" -I -m pip install 'torch==2.2.2' --index-url https://download.pytorch.org/whl/cpu
# Resolve platform-specific transitive wheels on Linux. The full lock snapshot
# was produced on macOS; this file pins the shared algorithm/runtime versions.
"$bench_python" -I -m pip install -r airl_runner/requirements.txt
"$bench_python" -I -m pip install --no-deps .
"$bench_python" -I -m pip check
MPLCONFIGDIR="$bench_root/.cache/matplotlib" "$bench_python" -I -c \
    'import sys, imitation, active_sensing_bench; from imitation.algorithms.adversarial.airl import AIRL; print("Python:", sys.executable); print("External imitation:", imitation.__file__); print("Benchmark:", active_sensing_bench.__file__)'

echo 'Setup complete. From this repository root, submit: sbatch scripts/irl_comparison.sbatch smoke'
