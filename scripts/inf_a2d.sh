python -m torch.distributed.launch --nproc_per_node=1 --use_env main.py \
--dataset_file a2d --with_box_refine --freeze_text_encoder --qtrans --f_token 8 --batch_size 16 \
--resume /home/xhu/Code/narval_ref/ytvos_dirs/r50_8ftoken_qtrains_5cocotdetr_finetune/checkpoint.pth \
--backbone resnet50  \
--eval