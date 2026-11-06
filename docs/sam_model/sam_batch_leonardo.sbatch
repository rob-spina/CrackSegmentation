#!/bin/bash
# SAM auto-segment for a whole CrackSegmentation data folder on one Leonardo (CINECA) GPU: sam_batch.py.
#
# Once, on the login node:
#   cd $WORK && git clone https://github.com/rob-spina/CrackSegmentation.git
#   source $WORK/cracksam_lora/venv/bin/activate          # the training venv: torch, torchvision, segment-anything
#   pip install opencv-python-headless scikit-image       # what the app needs on top
#   mkdir -p $WORK/drone_4k/Images                         # then copy the photos there (rsync from the Mac)
#
# Submit:      sbatch docs/sam_model/sam_batch_leonardo.sbatch
# Pilot run:   LIMIT=30 sbatch docs/sam_model/sam_batch_leonardo.sbatch
# Maps only:   PROB_ONLY=1 sbatch docs/sam_model/sam_batch_leonardo.sbatch   (for [SAM click to accept]; no JSON)
# Check the account / partition names against your other jobs (submit_leonardo.sbatch).
#SBATCH --job-name=sam_batch
#SBATCH --account=IscrC_AID-ECHO
#SBATCH --partition=boost_usr_prod
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=sam_batch_%j.out

set -euo pipefail

DATA_DIR=${DATA_DIR:-$WORK/drone_4k}                              # contains Images/; JSON files/ and Binary files/ are written
CODE_DIR=${CODE_DIR:-$WORK/CrackSegmentation}
WEIGHTS=${WEIGHTS:-$WORK/cracksam_lora}
SAM_BASE=${SAM_BASE:-$WEIGHTS/sam_vit_h_4b8939.pth}
CRACK_WEIGHTS=${CRACK_WEIGHTS:-$WEIGHTS/runs/facade390_vit_h_fixed/best_lora.pth}
LIMIT=${LIMIT:-0}                                                 # 0 = every photo
PROB_ONLY=${PROB_ONLY:-}                                          # 1 = crack maps only

module load python/3.11.7
source $WORK/cracksam_lora/venv/bin/activate
cd "$CODE_DIR"

python -u sam_batch.py \
    --data-dir "$DATA_DIR" \
    --sam-checkpoint "$SAM_BASE" \
    --lora-weights "$CRACK_WEIGHTS" \
    --model-type vit_h --lora-r 4 \
    --tile 512 --overlap 128 --batch-size 8 --device cuda \
    --limit "$LIMIT" ${PROB_ONLY:+--prob-only}
