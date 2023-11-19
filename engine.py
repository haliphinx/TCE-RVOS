"""
Train and eval functions used in main.py
Modified from DETR (https://github.com/facebookresearch/detr)
"""
import math
from models import postprocessors
import os
import sys
from typing import Iterable
from pathlib import Path
import numpy as np
import json
from PIL import Image
from tqdm import tqdm

import torch
import torch.distributed as dist
import torchvision.transforms as T
import torch.nn.functional as F

import util.misc as utils
from datasets.coco_eval import CocoEvaluator
from datasets.refexp_eval import RefExpEvaluator

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from datasets.a2d_eval import calculate_precision_at_k_and_iou_metrics
from models.segmentation import (dice_loss, sigmoid_focal_loss)

transform = T.Compose([
    T.Resize(360),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

def train_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, lr_scheduler = None, max_norm: float = 0):
    model.train()
    criterion.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.7f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 10
    for samples, targets in metric_logger.log_every(data_loader, print_freq, header):
        samples = samples.to(device)
        captions = [t["caption"] for t in targets]
        targets = utils.targets_to(targets, device) 

        outputs = model(samples, captions, targets)

        
        loss_dict = criterion(outputs, targets)

        weight_dict = criterion.weight_dict
        # print(loss_dict,weight_dict)
        losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        loss_dict_reduced_unscaled = {f'{k}_unscaled': v
                                      for k, v in loss_dict_reduced.items()}
        loss_dict_reduced_scaled = {k: v * weight_dict[k]
                                    for k, v in loss_dict_reduced.items() if k in weight_dict}
        # print("-_______________",loss_dict_reduced_scaled)
        losses_reduced_scaled = sum(loss_dict_reduced_scaled.values())
        # print("++++++++++++++++++",losses_reduced_scaled)
        loss_value = losses_reduced_scaled.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            print(loss_dict_reduced)
            sys.exit(1)
        optimizer.zero_grad()
        losses.backward()
        if max_norm > 0:
            grad_total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
        else:
            grad_total_norm = utils.get_total_grad_norm(model.parameters(), max_norm)
        # for name, param in model.named_parameters():
        #     if param.grad is None:
        #         print(name)
        optimizer.step()
        if lr_scheduler is not None:
            lr_scheduler.step()

        metric_logger.update(loss=loss_value, **loss_dict_reduced_scaled, **loss_dict_reduced_unscaled)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        metric_logger.update(grad_norm=grad_total_norm)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate(model, criterion, postprocessors, data_loader, evaluator_list, device, args):
    model.eval()
    criterion.eval()

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'

    for samples, targets in metric_logger.log_every(data_loader, 10, header):
        samples = samples.to(device)
        captions = [t["caption"] for t in targets]
        targets = utils.targets_to(targets, device)

        outputs = model(samples, captions, targets)
        loss_dict = criterion(outputs, targets)
        weight_dict = criterion.weight_dict

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        loss_dict_reduced_scaled = {k: v * weight_dict[k]
                                    for k, v in loss_dict_reduced.items() if k in weight_dict}
        loss_dict_reduced_unscaled = {f'{k}_unscaled': v
                                      for k, v in loss_dict_reduced.items()}
        metric_logger.update(loss=sum(loss_dict_reduced_scaled.values()),
                             **loss_dict_reduced_scaled,
                             **loss_dict_reduced_unscaled)

        orig_target_sizes = torch.stack([t["orig_size"] for t in targets], dim=0)
        results = postprocessors['bbox'](outputs, orig_target_sizes)
        if 'segm' in postprocessors.keys():
            target_sizes = torch.stack([t["size"] for t in targets], dim=0)
            results = postprocessors['segm'](results, outputs, orig_target_sizes, target_sizes)
        res = {target['image_id'].item(): output for target, output in zip(targets, results)}
        
        for evaluator in evaluator_list:
            evaluator.update(res)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    for evaluator in evaluator_list:
        evaluator.synchronize_between_processes()

    # accumulate predictions from all images
    refexp_res = None
    for evaluator in evaluator_list:
        if isinstance(evaluator, CocoEvaluator):
            evaluator.accumulate()
            evaluator.summarize()
        elif isinstance(evaluator, RefExpEvaluator):
            refexp_res = evaluator.summarize()

    stats = {k: meter.global_avg for k, meter in metric_logger.meters.items()}

    # update stats
    for evaluator in evaluator_list:
        if isinstance(evaluator, CocoEvaluator):
            if "bbox" in postprocessors.keys():
                stats["coco_eval_bbox"] = evaluator.coco_eval["bbox"].stats.tolist()
            if "segm" in postprocessors.keys():
                stats["coco_eval_masks"] = evaluator.coco_eval["segm"].stats.tolist()
    if refexp_res is not None:
        stats.update(refexp_res)
        
    return stats

@torch.no_grad()
def evaluate_yvos(model, args):
    model.eval()
    split = args.split
    # save path
    output_dir = args.output_dir
    save_path_prefix = os.path.join(output_dir, split)
    if not os.path.exists(save_path_prefix):
        os.makedirs(save_path_prefix)

    save_visualize_path_prefix = os.path.join(output_dir, split + '_images')
    if args.visualize:
        if not os.path.exists(save_visualize_path_prefix):
            os.makedirs(save_visualize_path_prefix)

    # load data
    root = Path(args.ytvos_path) # data/ref-youtube-vos
    img_folder = os.path.join(root, "train", "JPEGImages")
    ann_folder = os.path.join(root, "train", "Annotations")
    meta_file = os.path.join(root, "meta_expressions", split, "meta_expressions.json")
    with open(meta_file, "r") as f:
        data = json.load(f)["videos"]
    valid_test_videos = set(data.keys())
    # for some reasons the competition's validation expressions dict contains both the validation (202) & 
    # test videos (305). so we simply load the test expressions dict and use it to filter out the test videos from
    # the validation expressions dict:
    test_meta_file = os.path.join(root, "meta_expressions", "test", "meta_expressions.json")
    with open(test_meta_file, 'r') as f:
        test_data = json.load(f)['videos']
    test_videos = set(test_data.keys())
    valid_videos = valid_test_videos - test_videos
    video_list = sorted([video for video in valid_videos])
    
    assert len(video_list) == 200, 'error: incorrect number of validation videos'
    result = []
    
    # 1. For each video
    for video in tqdm(video_list,"Evaluate the model"):
        metas = [] # list[dict], length is number of expressions

        expressions = data[video]["expressions"]   
        expression_list = list(expressions.keys()) 
        num_expressions = len(expression_list)
        video_len = len(data[video]["frames"])

        # read all the anno meta
        for i in range(num_expressions):
            meta = {}
            meta["video"] = video
            meta["exp"] = expressions[expression_list[i]]["exp"]
            meta["obj_id"] = expressions[expression_list[i]]["obj_id"]
            meta["exp_id"] = expression_list[i]
            meta["frames"] = data[video]["frames"]
            metas.append(meta)
        meta = metas

        # 2. For each expression
        for i in range(num_expressions):
            video_name = meta[i]["video"]
            exp = meta[i]["exp"]
            # exp_id = meta[i]["exp_id"]
            frames = meta[i]["frames"]
            obj_id = int(meta[i]["obj_id"])
            

            video_len = len(frames)
            # store images
            imgs = []
            gt = []
            valids = []
            for t in range(video_len):
                frame = frames[t]
                img_path = os.path.join(img_folder, video_name, frame + ".jpg")
                img = Image.open(img_path).convert('RGB')
                gt_path = os.path.join(ann_folder, video_name, frame + ".png")
                gt_mask = np.array(Image.open(gt_path).convert('P'))
                gt_mask = (gt_mask==obj_id)
                if (gt_mask > 0).any():
                    valids.append(1)
                else:
                    valids.append(0)
                origin_w, origin_h = img.size
                imgs.append(transform(img)) # list[img]
                gt.append(gt_mask)
            gt = torch.Tensor(np.asarray(gt)).to(args.device)
            
            imgs = torch.stack(imgs, dim=0).to(args.device) # [video_len, 3, h, w]
            img_h, img_w = imgs.shape[-2:]
            size = torch.as_tensor([int(img_h), int(img_w)]).to(args.device)
            target = {"size": size}

            with torch.no_grad():
                outputs = model([imgs], [exp], [target])
            
            pred_logits = outputs["pred_logits"][0] 
            # pred_boxes = outputs["pred_boxes"][0]   
            pred_masks = outputs["pred_masks"][0]   
            # pred_ref_points = outputs["reference_points"][0]  

            # according to pred_logits, select the query index
            pred_scores = pred_logits.sigmoid() # [t, q, k]
            pred_scores = pred_scores.mean(0)   # [q, k]
            max_scores, _ = pred_scores.max(-1) # [q,]
            _, max_ind = max_scores.max(-1)     # [1,]
            max_inds = max_ind.repeat(video_len)
            pred_masks = pred_masks[range(video_len), max_inds, ...] # [t, h, w]
            pred_masks = pred_masks.unsqueeze(0)

            pred_masks = F.interpolate(pred_masks, size=(origin_h, origin_w), mode='bilinear', align_corners=False).squeeze(0)

            for t in range(video_len):
                if (gt[t]> 0).any():

                    src_masks = pred_masks.flatten(1) # [b, thw]

                    target_masks = gt.flatten(1) # [b, thw]
                    num_boxes = np.count_nonzero(valids)
                    dice = dice_loss(src_masks, target_masks, num_boxes).detach().cpu().tolist()
                    # dice = 0
                    sigmoid = sigmoid_focal_loss(src_masks, target_masks, num_boxes).detach().cpu().tolist()
                    # print(sigmoid)
                    result.append([dice,sigmoid])
    dice, sigmond = np.mean(result, axis=0)
    return dice, sigmond

class BytesEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, bytes):
            return str(obj, encoding='utf-8')
        return json.JSONEncoder.default(self, obj)

@torch.no_grad()
def evaluate_a2d(model, data_loader, postprocessor, device, args):
    model.eval()
    predictions = []
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'

    for samples, targets in metric_logger.log_every(data_loader, 10, header):
        image_ids = [t['image_id'] for t in targets]

        samples = samples.to(device)
        captions = [t["caption"] for t in targets]
        targets = utils.targets_to(targets, device)

        outputs = model(samples, captions, targets)

        orig_target_sizes = torch.stack([t["orig_size"] for t in targets], dim=0)
        target_sizes = torch.stack([t["size"] for t in targets], dim=0)
        processed_outputs = postprocessor(outputs, orig_target_sizes, target_sizes)

        for p, image_id in zip(processed_outputs, image_ids):
            for s, m in zip(p['scores'], p['rle_masks']):
                    predictions.append({'image_id': image_id,
                                        'category_id': 1,  # dummy label, as categories are not predicted in ref-vos
                                        'segmentation': m,
                                        'score': s.item()})

    # gather and merge predictions from all gpus
    gathered_pred_lists = utils.all_gather(predictions)
    predictions = [p for p_list in gathered_pred_lists for p in p_list]
    # print(predictions[0]['segmentation']['counts'])
    # save_path = "/home/xhu/Code/a2d_ana/test.json"
    # with open(save_path, 'w') as f:
    #     json.dump(predictions, f, cls=BytesEncoder)
        # np.save(f,predictions)
    # assert False
    # evaluation
    eval_metrics = {}
    if utils.is_main_process():
        if args.dataset_file == 'a2d':
            # coco_gt = COCO(os.path.join(args.a2d_path, 'a2d_sentences_test_annotations_in_coco_format.json'))
            coco_gt = COCO(os.path.join(args.a2d_path, 'a2d_sentences_test_annotations_in_coco_format_short.json'))
        elif args.dataset_file == 'jhmdb':
            coco_gt = COCO(os.path.join(args.jhmdb_path, 'jhmdb_sentences_gt_annotations_in_coco_format.json'))
        else:
            raise NotImplementedError
        coco_pred = coco_gt.loadRes(predictions)
        coco_eval = COCOeval(coco_gt, coco_pred, iouType='segm')
        coco_eval.params.useCats = 0  # ignore categories as they are not predicted in ref-vos task
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()
        ap_labels = ['mAP 0.5:0.95', 'AP 0.5', 'AP 0.75', 'AP 0.5:0.95 S', 'AP 0.5:0.95 M', 'AP 0.5:0.95 L']
        ap_metrics = coco_eval.stats[:6]
        eval_metrics = {l: m for l, m in zip(ap_labels, ap_metrics)}
        # Precision and IOU
        precision_at_k, overall_iou, mean_iou = calculate_precision_at_k_and_iou_metrics(coco_gt, coco_pred)
        eval_metrics.update({f'P@{k}': m for k, m in zip([0.5, 0.6, 0.7, 0.8, 0.9], precision_at_k)})
        eval_metrics.update({'overall_iou': overall_iou, 'mean_iou': mean_iou})
        print(eval_metrics)

    # sync all processes before starting a new epoch or exiting
    dist.barrier()
    return eval_metrics








