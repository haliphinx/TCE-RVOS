#!/usr/bin/env bash
set -x

GPUS=${GPUS:-1}
if [ $GPUS -lt 8 ]; then
    GPUS_PER_NODE=${GPUS_PER_NODE:-$GPUS}
else
    GPUS_PER_NODE=${GPUS_PER_NODE:-8}
fi
CPUS_PER_TASK=${CPUS_PER_TASK:-5}

# OUTPUT_DIR=/home/xhu/Code/ReferFormer/ytvos_dirs/r50_pre_connect_all_video
OUTPUT_DIR=ytvos_dirs/${1}
PRETRAINED_WEIGHTS=ckp/r50_pretrained.pth
# PRETRAINED_WEIGHTS=ckp/video_swin_tiny_pretrained.pth
PY_ARGS=${@:2}  # Any arguments from the forth one are captured by this

echo "Load pretrained weights from: ${PRETRAINED_WEIGHTS}"

# train
# PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
# python3 -m torch.distributed.launch --nproc_per_node=${GPUS_PER_NODE}  \
# --master_port=${PORT} --use_env \
# main.py --with_box_refine --freeze_text_encoder --binary \
# --epochs 9 --lr_drop 3 6  \
# --output_dir=${OUTPUT_DIR} --resume=${PRETRAINED_WEIGHTS} --backbone resnet50 --masks --email --start_epoch 7
PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
python -m torch.distributed.launch --nproc_per_node=${GPUS_PER_NODE} --use_env \
main_joint.py --with_box_refine --binary --lr 1e-4 --f_extra 1 \
--epochs 12 --lr_drop 8 10 --pretrain_coco \
--output_dir=${OUTPUT_DIR} --backbone x3d_self ${PY_ARGS}
# --resume --freeze_text_encoder  --lr_drop 4 6 --pretrained_weights=${PRETRAINED_WEIGHTS}
# # inference
# CHECKPOINT=${OUTPUT_DIR}/checkpoint.pth
# python3 inference_ytvos.py --with_box_refine --binary --freeze_text_encoder \
# --output_dir=${OUTPUT_DIR} --resume=${CHECKPOINT}  --backbone resnet50 video_swin_b_p4w7 video_swin_t_p4w7 x3d_self

# echo "Working path is: ${OUTPUT_DIR}"

