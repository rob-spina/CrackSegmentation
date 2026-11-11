#!/bin/bash
# active_learning_round_leonardo.sbatch -- CrackSegmentation 2.0.1 [Active learning], one round on Leonardo:
#   1. exports the labelled photos (Training + Validation set) for train.py   (active_learning_prep.py)
#   2. fine-tunes the model again, starting from the previous round's best_lora.pth  (cracksam_lora train.py)
#   3. puts the new best_lora.pth in  $DATA_DIR/SAM model/rounds/round_NN/
#   4. computes the new model's crack maps of the photos still to label and of the Test set  (sam_batch.py)
# Then copy back to the Mac (rsync) 'SAM model/rounds/round_NN' and 'SAM model/probability maps/<new tag>' and press
# [Active learning] in CrackSegmentation.
#
# Usage (first round = domain adaptation from the Facade390 model):
#   ROUND=1 PREV=$WORK/cracksam_lora/runs/facade390_vit_h_fixed/best_lora.pth \
#       sbatch docs/sam_model/active_learning_round_leonardo.sbatch
# next rounds:
#   ROUND=2 PREV="$WORK/drone_4k/SAM model/rounds/round_01/best_lora.pth" sbatch docs/sam_model/active_learning_round_leonardo.sbatch
# maps only (no training): the crack maps of EVERY photo (Images, Training / Validation / Test set, labelled ones
# too) with the PREV weights, e.g. x2 maps of the current model to label with:
#   MAPS_ONLY=1 UPSCALE=2 PREV=$WORK/cracksam_lora/runs/facade390_vit_h_fixed/best_lora.pth \
#       sbatch docs/sam_model/active_learning_round_leonardo.sbatch
#   -> 'SAM model/probability maps/lora-<sha of PREV>-x2'; on the Mac set SAM_UPSCALE = 2 in config.py
#      (weights without train_args.json are x1 for the app).
#
# train.py / dataset.py / losses.py / model.py in $LORA_DIR must be the CrackSegmentation 2.0.1 versions
# (--ignore-uncertain, EXIF-upright photos, --upscale): copy them from the Mac's echo-twin folder once.
# UPSCALE=2 (default): each 512 tile is enlarged to 1024 before SAM, in training and in the maps; the maps go to
# 'probability maps/lora-<sha>-x2'. Copy back round_NN/ whole (best_lora.pth + train_args.json).
# Before: rsync the data folder (with Training set / Validation set / Test set / Images / JSON files / Binary files)
# to $DATA_DIR, and `git pull` CrackSegmentation in $CODE_DIR.

#SBATCH --job-name=crackseg_al_round
#SBATCH --account=IscrC_AID-ECHO
#SBATCH --partition=boost_usr_prod
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

set -euo pipefail

MAPS_ONLY=${MAPS_ONLY:-0}
PREV=${PREV:?set PREV to the previous round best_lora.pth}
if [ "$MAPS_ONLY" = "1" ]; then ROUND=${ROUND:-0}; else ROUND=${ROUND:?set ROUND=1, 2, ...}; fi
DATA_DIR=${DATA_DIR:-$WORK/drone_4k}
CODE_DIR=${CODE_DIR:-$WORK/CrackSegmentation}
LORA_DIR=${LORA_DIR:-$WORK/cracksam_lora}                         # train.py, model.py ... of cracksam_lora
SAM_BASE=${SAM_BASE:-$LORA_DIR/sam_vit_h_4b8939.pth}
EPOCHS=${EPOCHS:-20}
LR=${LR:-2e-5}
TILES=${TILES:-8}                                                 # random 512 tiles per photo per epoch
UPSCALE=${UPSCALE:-2}                     # 512 tiles enlarged x2 before SAM (thin cracks); UPSCALE=1 = as before
ROUND_NAME=$(printf "round_%02d" "$ROUND")
EXPORT_DIR=$LORA_DIR/al_data/$ROUND_NAME
RUN_DIR=$LORA_DIR/runs/al_$ROUND_NAME
ROUNDS_DIR="$DATA_DIR/SAM model/rounds/$ROUND_NAME"

module load python/3.11.7
source $LORA_DIR/venv/bin/activate

if [ "$MAPS_ONLY" = "1" ]; then
    echo "== maps only: every photo, weights $PREV, upscale x$UPSCALE"
    cd "$CODE_DIR"
    for SUB in "Test set/Images" "Images" "Training set/Images" "Validation set/Images"; do
        [ -d "$DATA_DIR/$SUB" ] || continue
        python -u sam_batch.py --data-dir "$DATA_DIR" --images-subdir "$SUB" --prob-only \
            --sam-checkpoint "$SAM_BASE" --lora-weights "$PREV" \
            --model-type vit_h --lora-r 4 --tile 512 --overlap 128 --upscale "$UPSCALE" --batch-size 8 --device cuda
    done
    echo "== done: copy back 'SAM model/probability maps' (Fetch results)"
    exit 0
fi

echo "== 1. export of the labelled photos"
python -u "$CODE_DIR/active_learning_prep.py" --data-dir "$DATA_DIR" --out "$EXPORT_DIR"

echo "== 2. fine-tuning from $PREV"
cd "$LORA_DIR"
python -u train.py \
    --data-root "$EXPORT_DIR" --image-ext .jpg --mask-ext .png \
    --sam-checkpoint "$SAM_BASE" --model-type vit_h --lora-r 4 \
    --init-lora-weights "$PREV" \
    --tile-size 512 --tiles-per-image "$TILES" --upscale "$UPSCALE" \
    --lr "$LR" --epochs "$EPOCHS" --val-split 0.15 --ignore-uncertain \
    --out-dir "$RUN_DIR"

echo "== 3. new weights -> $ROUNDS_DIR"
mkdir -p "$ROUNDS_DIR"
cp "$RUN_DIR/best_lora.pth" "$ROUNDS_DIR/best_lora.pth"
cp "$RUN_DIR/train_args.json" "$ROUNDS_DIR/train_args.json"     # CrackSegmentation reads the upscale from it
printf "round %s\nfrom %s\nphotos %s\nupscale %s\ndate %s\n" "$ROUND" "$PREV" "$(ls "$EXPORT_DIR/images" | wc -l)" \
    "$UPSCALE" "$(date -I)" \
    > "$ROUNDS_DIR/round_info.txt"

echo "== 4. crack maps of the new model"
cd "$CODE_DIR"
for SUB in "Images" "Test set/Images"; do
    [ -d "$DATA_DIR/$SUB" ] || continue
    SKIP=""; [ "$SUB" = "Images" ] && SKIP="--skip-labelled"     # the Test set photos are labelled: map them all
    python -u sam_batch.py --data-dir "$DATA_DIR" --images-subdir "$SUB" --prob-only $SKIP \
        --sam-checkpoint "$SAM_BASE" --lora-weights "$ROUNDS_DIR/best_lora.pth" \
        --model-type vit_h --lora-r 4 --tile 512 --overlap 128 --upscale "$UPSCALE" --batch-size 8 --device cuda
done
echo "== done: copy back 'SAM model/rounds/$ROUND_NAME' and 'SAM model/probability maps' and press Active learning"
