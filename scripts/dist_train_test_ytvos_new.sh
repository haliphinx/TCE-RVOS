#!/usr/bin/env bash
set -x
which python
ngpu=1
GPUS=${GPUS:-$ngpu}
if [ $GPUS -lt 8 ]; then
    GPUS_PER_NODE=${GPUS_PER_NODE:-$GPUS}
else
    GPUS_PER_NODE=${GPUS_PER_NODE:-8}
fi
CPUS_PER_TASK=${CPUS_PER_TASK:-5}

# OUTPUT_DIR=/home/xhu/Code/ReferFormer/ytvos_dirs/r50_pre_connect_all_video
OUTPUT_DIR=ytvos_dirs/${1}
# PRETRAINED_WEIGHTS=/home/xhu/Code/ReferFormer_update/ckp/pretrained/r50_pretrained.pth
PRETRAINED_WEIGHTS=/home/xhu/Code/ReferFormer_update/ckp/trained/r50_query_eachlayer_best/checkpoint.pth

# PRETRAINED_WEIGHTS=ckp/X3D_S.pyth
# PRETRAINED_WEIGHTS=/home/xhu/Desktop/checkpoint.pth
PY_ARGS=${@:2}  # Any arguments from the forth one are captured by this

echo "Load pretrained weights from: ${PRETRAINED_WEIGHTS}"

# python inference_ytvos.py --with_box_refine --binary --masks --freeze_text_encoder --ngpu ${ngpu} --keep_fps --num_frames 1 --qtrans \
# --output_dir=${OUTPUT_DIR} \
# --resume=${PRETRAINED_WEIGHTS} \
# --backbone resnet50 ${PY_ARGS}

# train 192881543, 191826823, 177226111, 176171391  --keep_fps --num_frames 5  --f_token 8
# PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
# python3 -m torch.distributed.launch --nproc_per_node=${GPUS_PER_NODE}  \
# --master_port=${PORT} --use_env \
# main.py --with_box_refine --freeze_text_encoder --binary \
# --epochs 9 --lr_drop 3 6  \
# --output_dir=${OUTPUT_DIR} --resume=${PRETRAINED_WEIGHTS} --backbone resnet50 --masks --email --start_epoch 7
# PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
python -m torch.distributed.launch --nproc_per_node=${GPUS_PER_NODE} --use_env \
main.py --with_box_refine --binary \
--epochs 7 --lr_drop 4 6 --pretrained_weights=${PRETRAINED_WEIGHTS} --f_token 8 --qtrans \
--output_dir=${OUTPUT_DIR} --backbone resnet50 ${PY_ARGS}
# --resume --freeze_text_encoder  --f_token 8 --qtrans --tdetr --pretrained_weights=${PRETRAINED_WEIGHTS}
# # inference --pretrained_weights=${PRETRAINED_WEIGHTS} 
# CHECKPOINT=${OUTPUT_DIR}/checkpoint.pth  --vlblock --cyclic_lr

# python3 inference_ytvos.py --with_box_refine --binary --freeze_text_encoder \
# --output_dir=${OUTPUT_DIR} --resume=${CHECKPOINT}  --backbone resnet50 video_swin_b_p4w7 x3d_xs x3d_self

# echo "Working path is: ${OUTPUT_DIR}"

# 178017389 177088101