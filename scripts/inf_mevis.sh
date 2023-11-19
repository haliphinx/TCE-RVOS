# OUTPUT_DIR=./ytvos_dirs/${1}
OUTPUT_DIR=./ytvos_dirs/test1

PY_ARGS=${@:2}
# PRETRAINED_WEIGHTS=${OUTPUT_DIR}/checkpoint0006.pth
PRETRAINED_WEIGHTS=ytvos_dirs/r50_8ftoken_qtrains_5cocotdetr_finetune/checkpoint.pth
# PRETRAINED_WEIGHTS=ckp/pretrained/r50_pretrained.pth

python inference_mevis.py --with_box_refine --binary --masks --freeze_text_encoder --ngpu 1 --qtrans --f_token 8 --keep_fps \
--output_dir=${OUTPUT_DIR} \
--resume=${PRETRAINED_WEIGHTS} \
--backbone resnet50 ${PY_ARGS}

