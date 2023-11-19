python eval_vidstg.py \
--with_box_refine \
--binary \
--freeze_text_encoder \
--output_dir=/home/xhu/Code/artificial_occlusion/test \
--resume=/home/xhu/Code/ReferFormer/ytvos_dirs/r50_from_pretrain_pre_frame_connect_skip_frame_vis_loss/checkpoint.pth  \
--backbone resnet50 \
--masks \
--vis_loss \
--visualize