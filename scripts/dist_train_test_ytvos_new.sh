#!/usr/bin/env bash
set -x
which python
ngpu=8
GPUS=${GPUS:-$ngpu}
if [ $GPUS -lt 8 ]; then
    GPUS_PER_NODE=${GPUS_PER_NODE:-$GPUS}
else
    GPUS_PER_NODE=${GPUS_PER_NODE:-8}
fi
CPUS_PER_TASK=${CPUS_PER_TASK:-5}

OUTPUT_DIR=ytvos_dirs/${1}
PRETRAINED_WEIGHTS=/ckp/trained/r50_query_eachlayer_best/checkpoint.pth

PY_ARGS=${@:2}  # Any arguments from the forth one are captured by this

echo "Load pretrained weights from: ${PRETRAINED_WEIGHTS}"

python -m torch.distributed.launch --nproc_per_node=${GPUS_PER_NODE} --use_env \
main.py --with_box_refine --binary \
--epochs 6 --lr_drop 3 5 --pretrained_weights=${PRETRAINED_WEIGHTS} --f_token 8 --qtrans \
--output_dir=${OUTPUT_DIR} --backbone resnet50 ${PY_ARGS}
