"""
Ref-YoutubeVOS data loader
"""
from pathlib import Path

import torch
from torch.autograd.grad_mode import F
from torch.utils.data import Dataset
import datasets.transforms_video as T

import os
from PIL import Image
import json
import numpy as np
import random

from datasets.categories import ytvos_category_dict as category_dict


class YTVOSDataset(Dataset):
    """
    A dataset class for the Refer-Youtube-VOS dataset which was first introduced in the paper:
    "URVOS: Unified Referring Video Object Segmentation Network with a Large-Scale Benchmark"
    (see https://link.springer.com/content/pdf/10.1007/978-3-030-58555-6_13.pdf).
    The original release of the dataset contained both 'first-frame' and 'full-video' expressions. However, the first
    dataset is not publicly available anymore as now only the harder 'full-video' subset is available to download
    through the Youtube-VOS referring video object segmentation competition page at:
    https://competitions.codalab.org/competitions/29139
    Furthermore, for the competition the subset's original validation set, which consists of 507 videos, was split into
    two competition 'validation' & 'test' subsets, consisting of 202 and 305 videos respectively. Evaluation can
    currently only be done on the competition 'validation' subset using the competition's server, as
    annotations were publicly released only for the 'train' subset of the competition.

    """
    def __init__(self, img_folder: Path, ann_file: Path, transforms, return_masks: bool, 
                 num_frames: int, max_skip: int, args = None):
        self.args = args
        self.img_folder = img_folder     
        self.ann_file = ann_file         
        self._transforms = transforms    
        self.return_masks = return_masks # not used
        self.num_frames = num_frames     
        self.max_skip = max_skip
        self.keep_fps = args.keep_fps
        self.f_extra = 0
        self.counter = -1
        # create video meta data
        self.prepare_metas()       

        print('\n video num: ', len(self.videos), ' clip num: ', len(self.metas), ' current counter: ', self.counter)  
        print('\n')    

    def refresh_metas(self):
        self.counter += 1
        self.counter %= self.num_frames
        self.prepare_metas()
        print('\n video num: ', len(self.videos), ' clip num: ', len(self.metas), ' current counter: ', self.counter)  

    def prepare_metas(self):
        # read object information
        with open(os.path.join(str(self.img_folder), 'meta.json'), 'r') as f:
            subset_metas_by_video = json.load(f)['videos']
        
        # read expression data
        with open(str(self.ann_file), 'r') as f:
            subset_expressions_by_video = json.load(f)['videos']
        self.videos = list(subset_expressions_by_video.keys())

        self.metas = []
        for vid in self.videos:
            vid_meta = subset_metas_by_video[vid]
            vid_data = subset_expressions_by_video[vid]
            vid_frames = sorted(vid_data['frames'])
            vid_len = len(vid_frames)
            for exp_id, exp_dict in vid_data['expressions'].items():
                if self.keep_fps:
                    for frame_id in range(self.counter, vid_len - self.num_frames, self.num_frames - 1):
                        meta = {}
                        meta['video'] = vid
                        meta['exp'] = exp_dict['exp']
                        meta['obj_id'] = int(exp_dict['obj_id'])
                        meta['frames'] = vid_frames
                        meta['frame_id'] = frame_id
                        # get object category
                        obj_id = exp_dict['obj_id']
                        meta['category'] = vid_meta['objects'][obj_id]['category']
                        self.metas.append(meta)
                else:
                    for frame_id in range(0, vid_len, self.num_frames):
                        meta = {}
                        meta['video'] = vid
                        meta['exp'] = exp_dict['exp']
                        meta['obj_id'] = int(exp_dict['obj_id'])
                        meta['frames'] = vid_frames
                        meta['frame_id'] = frame_id
                        # get object category
                        obj_id = exp_dict['obj_id']
                        meta['category'] = vid_meta['objects'][obj_id]['category']
                        self.metas.append(meta)


    @staticmethod
    def bounding_box(img):
        rows = np.any(img, axis=1)
        cols = np.any(img, axis=0)
        rmin, rmax = np.where(rows)[0][[0, -1]]
        cmin, cmax = np.where(cols)[0][[0, -1]]
        return rmin, rmax, cmin, cmax # y1, y2, x1, x2 
        
    def __len__(self):
        return len(self.metas)
        
    def __getitem__(self, idx):
        instance_check = False
        while not instance_check:
            meta = self.metas[idx]  # dict

            video, exp, obj_id, category, frames, frame_id = \
                        meta['video'], meta['exp'], meta['obj_id'], meta['category'], meta['frames'], meta['frame_id']
            # clean up the caption
            exp = " ".join(exp.lower().split())
            category_id = category_dict[category]
            vid_len = len(frames)

            num_frames = self.num_frames + 2 * self.f_extra
            # random sparse sample
            sample_indx = [frame_id]
            if self.args.vid_aug:
                valid_fps = min((vid_len - frame_id) // num_frames, 3)
                step = random.randint(1,valid_fps)
            else:
                step = 1
            if self.keep_fps:
                if num_frames != 1:
                    for sampled_id in range(1, self.num_frames * step, step):
                        assert (frame_id + sampled_id) < vid_len, "Frame sampled out of the range."
                        sample_indx.append(frame_id + sampled_id) 
            else:
                if num_frames != 1:
                    # local sample
                    sample_id_before = random.randint(1, 3)
                    sample_id_after = random.randint(1, 3)
                    local_indx = [max(0, frame_id - sample_id_before), min(vid_len - 1, frame_id + sample_id_after)]
                    sample_indx.extend(local_indx)
        
                    # global sampling
                    if num_frames > 3:
                        all_inds = list(range(vid_len))
                        global_inds = all_inds[:min(sample_indx)] + all_inds[max(sample_indx):]
                        global_n = num_frames - len(sample_indx)
                        if len(global_inds) > global_n:
                            select_id = random.sample(range(len(global_inds)), global_n)
                            for s_id in select_id:
                                sample_indx.append(global_inds[s_id])
                        elif vid_len >=global_n:  # sample long range global frames
                            select_id = random.sample(range(vid_len), global_n)
                            for s_id in select_id:
                                sample_indx.append(all_inds[s_id])
                        else:
                            select_id = random.sample(range(vid_len), global_n - vid_len) + list(range(vid_len))           
                            for s_id in select_id:                                                                   
                                sample_indx.append(all_inds[s_id])
            sample_indx.sort()
            if self.args.vid_aug and (np.random.rand() < 0.5):
                sample_indx.reverse()
            # read frames and masks
            imgs, labels, boxes, masks, valid = [], [], [], [], []
            for j in range(num_frames):
                frame_indx = sample_indx[j]
                frame_name = frames[frame_indx]
                img_path = os.path.join(str(self.img_folder), 'JPEGImages', video, frame_name + '.jpg')
                mask_path = os.path.join(str(self.img_folder), 'Annotations', video, frame_name + '.png')
                img = Image.open(img_path).convert('RGB')
                if self.args.vid_aug and (np.random.rand() < 1/num_frames):
                    img = random_mask(img)
                mask = Image.open(mask_path).convert('P')

                # create the target
                label =  torch.tensor(category_id) 
                mask = np.array(mask)
                mask = (mask==obj_id).astype(np.float32) # 0,1 binary
                if (mask > 0).any():
                    y1, y2, x1, x2 = self.bounding_box(mask)
                    box = torch.tensor([x1, y1, x2, y2]).to(torch.float)
                    valid.append(1)
                else: # some frame didn't contain the instance
                    box = torch.tensor([0, 0, 0, 0]).to(torch.float) 
                    valid.append(0)
                    # img.save("/home/xhu/Code/ReferFormer/ytvos_dirs/test/image_text.jpg")
                    # print(exp)
                    # print(img_path)
                    # assert False
                mask = torch.from_numpy(mask)

                # append
                imgs.append(img)
                labels.append(label)
                masks.append(mask)
                boxes.append(box)

            # transform
            w, h = img.size
            labels = torch.stack(labels, dim=0) 
            boxes = torch.stack(boxes, dim=0) 
            boxes[:, 0::2].clamp_(min=0, max=w)
            boxes[:, 1::2].clamp_(min=0, max=h)
            masks = torch.stack(masks, dim=0) 
            target = {
                'frames_idx': torch.tensor(sample_indx), # [T,]
                'labels': labels,                        # [T,]
                'boxes': boxes,                          # [T, 4], xyxy
                'masks': masks,                          # [T, H, W]
                'valid': torch.tensor(valid),            # [T,]
                'caption': exp,
                'orig_size': torch.as_tensor([int(h), int(w)]), 
                'size': torch.as_tensor([int(h), int(w)]),
                'instance_check': torch.tensor(0)
            }

            # "boxes" normalize to [0, 1] and transform from xyxy to cxcywh in self._transform
            imgs, target = self._transforms(imgs, target) 
            imgs = torch.stack(imgs, dim=0) # [T, 3, H, W]
            
            # FIXME: handle "valid", since some box may be removed due to random crop
            # instance_check = True
            if torch.any(target['valid'] == 1):  # at leatst one instance
                instance_check = True
                target['instance_check'] = torch.tensor(1)
            else:
                idx = random.randint(0, self.__len__() - 1)

        return imgs, clip_target(target, self.f_extra)

def clip_target(target, f_extra):
    '''
    Clip out the target for extra frames to fit the citation function

    '''
    if f_extra == 0:
        return target
    target['frames_idx'] = target['frames_idx'][f_extra:-f_extra]
    target['labels'] = target['labels'][f_extra:-f_extra]
    target['boxes'] = target['boxes'][f_extra:-f_extra]
    target['masks'] = target['masks'][f_extra:-f_extra]
    target['valid'] = target['valid'][f_extra:-f_extra]

    return target


def make_coco_transforms(image_set, max_size=640):
    normalize = T.Compose([
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    scales = [288, 320, 352, 392, 416, 448, 480, 512]

    if image_set == 'train':
        return T.Compose([
            T.RandomHorizontalFlip(),
            T.PhotometricDistort(),
            T.RandomSelect(
                T.Compose([
                    T.RandomResize(scales, max_size=max_size),
                    T.Check(),
                ]),
                T.Compose([
                    T.RandomResize([400, 500, 600]),
                    T.RandomSizeCrop(384, 600),
                    T.RandomResize(scales, max_size=max_size),
                    T.Check(),
                ])
            ),
            normalize,
        ])
    
    # we do not use the 'val' set since the annotations are inaccessible
    if image_set == 'val':
        return T.Compose([
            T.RandomResize([360], max_size=640),
            normalize,
        ])

    raise ValueError(f'unknown {image_set}')

def random_mask(img, ratio = 0.5):
    w,h = img.size
    xc = random.random() * (w - 1)
    yc = random.random() * (h - 1)
    h_r = random.random() * (h - 1) * ratio
    w_r = random.random() * (w - 1) * ratio
    xl = int(max(0, xc - w_r/2))
    xh = int(min(w, xc + w_r/2))
    hl = int(max(0, yc - h_r/2))
    hh = int(min(h, yc + h_r/2))
    for i in range(xl, xh):
        for j in range(hl, hh):
            img.putpixel((i,j),(0,0,0))
    return img
    

def build(image_set, args, keep_fps = False):
    root = Path(args.ytvos_path)
    assert root.exists(), f'provided YTVOS path {root} does not exist'
    PATHS = {
        "train": (root / "train", root / "meta_expressions" / "train" / "meta_expressions.json"),
        "val": (root / "valid", root / "meta_expressions" / "val" / "meta_expressions.json"),    # not used actually
    }
    img_folder, ann_file = PATHS[image_set]
    dataset = YTVOSDataset(img_folder, ann_file, transforms=make_coco_transforms(image_set, max_size=args.max_size), return_masks=args.masks, 
                           num_frames=args.num_frames, max_skip=args.max_skip, args = args)
    return dataset

