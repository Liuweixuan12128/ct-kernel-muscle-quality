#!/bin/bash
#SBATCH --job-name=srepsegH
#SBATCH --partition=h100
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=64
#SBATCH --mem=256G
#SBATCH --time=06:00:00
#SBATCH --output=logs/srep_seg_sharp_h100.%j.out
# Same lock-file workers as srep_seg_sharp.sbatch, two per H100.
cd "${CT_KERNEL_PROJECT_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
mkdir -p logs
for g in 0 1 2 3 4 5 6 7; do for w in 0 1; do
  CUDA_VISIBLE_DEVICES=$g ./env/bin/python pipeline/p2_segment_worker.py --nii-dir data/nifti/nlst_pairs_sharp \
    --seg-dir data/seg/nlst_pairs_sharp --done-flag logs/srep_seg_sharp.done > logs/srep_seg_sharp.h${g}.${w}.log 2>&1 &
  sleep 5
done; done
wait
