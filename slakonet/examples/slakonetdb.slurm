#!/bin/bash
#SBATCH --job-name=sldb
#SBATCH --partition=main
#SBATCH --cpus-per-task=8
#SBATCH --time=12:00:00
#SBATCH --output=/data/kchoudh2/work/slakonetdb/logs/%x_%A_%a.out
#SBATCH --error=/data/kchoudh2/work/slakonetdb/logs/%x_%A_%a.err
set -euo pipefail
# Pin threads: torch and BLAS otherwise each grab all 256 cores and the
# tasks thrash against each other (a 64-k eigensolve once spun for half an
# hour at 750% CPU from exactly this).
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8
export PYTHONPATH=/data/kchoudh2/Software/env_agapi/slakonet
PY=/data/kchoudh2/mf_x86_64/envs/sldb/bin/python
echo "host=$(hostname) shard=$SLURM_ARRAY_TASK_ID/$SLURM_ARRAY_TASK_COUNT start=$(date)"
$PY -u /data/kchoudh2/work/slakonetdb/runs/slakonetdb_run.py \
    --shard "$SLURM_ARRAY_TASK_ID" --nshards "$SLURM_ARRAY_TASK_COUNT" \
    --out /data/kchoudh2/work/slakonetdb/out --device cpu
echo "done=$(date)"
