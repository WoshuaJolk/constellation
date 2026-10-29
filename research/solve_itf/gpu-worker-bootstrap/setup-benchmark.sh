#!/bin/bash
set -euo pipefail
cd /workspace/constellation
tar -xzf worker-payload.tar.gz
export DEBIAN_FRONTEND=noninteractive
timeout 180 apt-get update -qq
timeout 180 apt-get install -y -qq python3-venv
python3 -m venv .venv
.venv/bin/python -m pip install --disable-pip-version-check 'numpy>=1.26,<3' 'scipy>=1.11,<2' 'numba>=0.60,<1' 'numba-cuda[cu13]'  'astropy>=6,<9' 'pytest>=8,<10' 'jsonschema>=4.21,<5'
export PYTHONPATH=/workspace/constellation/src
export NUMBA_CACHE_DIR=/workspace/constellation/.numba-cache
export OPENBLAS_NUM_THREADS=1
.venv/bin/python - <<'PY'
import json,subprocess,os,platform
from importlib.metadata import version
from pathlib import Path
r=dict(python=platform.python_version(),cpus=os.cpu_count(),packages={p:version(p) for p in ['numpy','scipy','numba','numba-cuda','llvmlite','astropy']},nvidia_smi=subprocess.check_output(['nvidia-smi','--query-gpu=name,driver_version,memory.total','--format=csv,noheader'],text=True))
Path('hardware.json').write_text(json.dumps(r,indent=2));print(json.dumps(r))
PY
.venv/bin/python -m pytest -q tests/test_itf_gpu.py -p no:cacheprovider > gpu-tests.log 2>&1
.venv/bin/python research/solve_itf/gpu_benchmark.py run --bank bank.npz --backend both --cpu-threads 1 --seconds 30 --batch-cases 1024 --output gpu-comparison.json
