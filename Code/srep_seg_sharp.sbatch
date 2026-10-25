#!/bin/bash
#SBATCH --job-name=srepseg
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --array=0-11
#SBATCH --time=06:00:00
#SBATCH --output=logs/srep_seg_sharp.%A_%a.out
# Sci Rep revision (review point 3): segment the sharp partner of every analysed NLST pair on its own
# (total, abdominal_muscles, body). Lock-file workers (p2_segment_worker.py) share one folder across nodes.
cd "${CT_KERNEL_PROJECT_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
mkdir -p logs
for w in 0 1; do
  ./env/bin/python pipeline/p2_segment_worker.py --nii-dir data/nifti/nlst_pairs_sharp --seg-dir data/seg/nlst_pairs_sharp \
    --done-flag logs/srep_seg_sharp.done > logs/srep_seg_sharp.w${SLURM_ARRAY_TASK_ID}.${w}.log 2>&1 &
  sleep 15
done
wait
