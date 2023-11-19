# OUTPUT_DIR=./ytvos_dirs/${1}
OUTPUT_DIR=./ytvos_dirs/test1

PY_ARGS=${@:2}
# PRETRAINED_WEIGHTS=${OUTPUT_DIR}/checkpoint0006.pth
PRETRAINED_WEIGHTS=ytvos_dirs/r50_8ftoken_qtrains_5cocotdetr_finetune/checkpoint.pth
# PRETRAINED_WEIGHTS=ckp/pretrained/r50_pretrained.pth

python inference_ytvos.py --with_box_refine --binary --masks --freeze_text_encoder --ngpu 1 --qtrans --f_token 8 \
--output_dir=${OUTPUT_DIR} \
--resume=${PRETRAINED_WEIGHTS} \
--backbone resnet50 ${PY_ARGS}

# --f_token 8 --qtrans

# mv ${OUTPUT_DIR}/valid/ ${OUTPUT_DIR}/Annotations/
# cd ${OUTPUT_DIR}
# zip -r submission6.zip Annotations

# rm -r Annotations
# cd ../../

# PRETRAINED_WEIGHTS=${OUTPUT_DIR}/checkpoint0005.pth

# python inference_ytvos.py --with_box_refine --binary --masks --freeze_text_encoder --ngpu 4 \
# --output_dir=${OUTPUT_DIR} \
# --resume=${PRETRAINED_WEIGHTS} \
# --backbone resnet50 ${PY_ARGS}

# mv ${OUTPUT_DIR}/valid/ ${OUTPUT_DIR}/Annotations/
# cd ${OUTPUT_DIR}
# zip -r submission5.zip Annotations

# rm -r Annotations
# cd ../../

# PRETRAINED_WEIGHTS=${OUTPUT_DIR}/checkpoint0004.pth

# python inference_ytvos.py --with_box_refine --binary --masks --freeze_text_encoder --ngpu 4 \
# --output_dir=${OUTPUT_DIR} \
# --resume=${PRETRAINED_WEIGHTS} \
# --backbone resnet50 ${PY_ARGS}

# mv ${OUTPUT_DIR}/valid/ ${OUTPUT_DIR}/Annotations/
# cd ${OUTPUT_DIR}
# zip -r submission4.zip Annotations

# rm -r Annotations
# cd ../../

# PRETRAINED_WEIGHTS=${OUTPUT_DIR}/checkpoint0003.pth

# python inference_ytvos.py --with_box_refine --binary --masks --freeze_text_encoder --ngpu 4 \
# --output_dir=${OUTPUT_DIR} \
# --resume=${PRETRAINED_WEIGHTS} \
# --backbone resnet50 ${PY_ARGS}

# mv ${OUTPUT_DIR}/valid/ ${OUTPUT_DIR}/Annotations/
# cd ${OUTPUT_DIR}
# zip -r submission3.zip Annotations

# rm -r Annotations
# cd ../../



# PRETRAINED_WEIGHTS=${OUTPUT_DIR}/checkpoint0002.pth

# python inference_ytvos.py --with_box_refine --binary --masks --freeze_text_encoder --ngpu 4 \
# --output_dir=${OUTPUT_DIR} \
# --resume=${PRETRAINED_WEIGHTS} \
# --backbone resnet50 ${PY_ARGS}

# mv ${OUTPUT_DIR}/valid/ ${OUTPUT_DIR}/Annotations/
# cd ${OUTPUT_DIR}
# zip -r submission2.zip Annotations

# rm -r Annotations
# cd ../../

# PRETRAINED_WEIGHTS=${OUTPUT_DIR}/checkpoint0001.pth

# python inference_ytvos.py --with_box_refine --binary --masks --freeze_text_encoder --ngpu 4 \
# --output_dir=${OUTPUT_DIR} \
# --resume=${PRETRAINED_WEIGHTS} \
# --backbone resnet50 ${PY_ARGS}

# mv ${OUTPUT_DIR}/valid/ ${OUTPUT_DIR}/Annotations/
# cd ${OUTPUT_DIR}
# zip -r submission1.zip Annotations

# rm -r Annotations
# cd ../../

# PRETRAINED_WEIGHTS=${OUTPUT_DIR}/checkpoint0000.pth

# python inference_ytvos.py --with_box_refine --binary --masks --freeze_text_encoder --ngpu 4 \
# --output_dir=${OUTPUT_DIR} \
# --resume=${PRETRAINED_WEIGHTS} \
# --backbone resnet50 ${PY_ARGS}

# mv ${OUTPUT_DIR}/valid/ ${OUTPUT_DIR}/Annotations/
# cd ${OUTPUT_DIR}
# zip -r submission0.zip Annotations

# rm -r Annotations
# cd ../../