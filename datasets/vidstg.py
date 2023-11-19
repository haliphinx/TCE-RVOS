'''
VidSTG dataloader
'''

from pathlib import Path
from unicodedata import category

import torch
from torch.autograd.grad_mode import F
from torch.utils.data import Dataset
import datasets.transforms_video as T

import os
from PIL import Image
import json
import numpy as np
import random
from tqdm import tqdm
import cv2

class VIDSTGDataset(Dataset):
    '''
    Vidstg dataset is a visual reference detection dataset, details can be found https://github.com/Guaranteer/VidSTG-Dataset
    '''
    def __init__(self, img_folder: Path, ann_file: Path, transforms, return_masks: bool, 
                 num_frames: int, max_skip: int, fps:int = 5):
        '''
        The dataloader for the vidstg dataset.
        Input:
            dataset_folder: the folder to the dataset, it should contain an ann folder with a video folder
            transforms: the transform used for data augmentation
            num_frames: the number of frames for each video item
            data_split: the split of the data to be loaded, should be one of the ["train","val","test"]
        '''
        self.num_frame = num_frames
        print(img_folder)
        self.img_folder = img_folder     
        self.ann_file = ann_file 
        self._transforms = transforms
        self.max_skip = max_skip
        self.fps = fps
        print("build vidstg dataset.")
        self.prepare_metas()

    def prepare_metas(self):
        # # Read the trajectory data
        # with open(str(self.ann_file[0]), 'r') as f:
        #     traj_meta = json.load(f)

        # Read the annotation data
        with open(str(self.ann_file[1]), 'r') as f:
            ann_meta = json.load(f)["videos"]

        self.metas = []

        # loop through all ann records
        for ann in tqdm(ann_meta, "Generating the meta data"):
            # Read the data from the json
            video_id = ann["original_video_id"]
            original_fps = ann["fps"]
            start_frame = ann["start_frame"]
            end_frame = ann["end_frame"]
            tube_start_frame = ann["tube_start_frame"]
            tube_end_frame = ann["tube_end_frame"]
            caption = ann["caption"]
            category = ann["type"]
            target_id = ann["target_id"] # for traj searching
            video_path = ann["video_path"]
            w = ann["width"]
            h = ann["height"]

            # Read the traj to the ann
            # traj = traj_meta[video_id]["trajectories"]

            rel_fps = int((original_fps+0.5) // self.fps) # the relative fps to down sampling the video
            num_frames = rel_fps * self.num_frame
            
            for f_id in range(start_frame, end_frame, num_frames):
                meta = {}
                meta["video"] = video_id
                meta["video_path"] = video_path
                meta["exp"] = caption
                meta["category"] = category
                meta["frames"] = [f_id + i * rel_fps for i in range(self.num_frame)]
                meta["boxes"] = []
                meta["valid"] = []
                # Load the bbox for each frame
                # for id in range(f_id, f_id+num_frames, rel_fps):
                #     if str(id) in traj and str(target_id) in traj[str(id)]:
                #         bbox = traj[str(id)][str(target_id)]["bbox"]
                #         bbox[2] += bbox[0]
                #         bbox[3] += bbox[1]
                #         valid = 1
                #     else:
                #         bbox = [0,0,0,0]
                #         valid = 0
                #     meta["boxes"].append(bbox)
                #     meta["valid"].append(valid)
                self.metas.append(meta)
                    



    def __len__(self):
        return len(self.metas)

    def __getitem__(self, idx):
        imgs = []
        target = {}
        meta = self.metas[idx]
        folder_id= meta["video_path"].split("/")[0]
        video_id = meta["video"]
        frame_folder = os.path.join(self.img_folder, folder_id, video_id)
        ann_path = os.path.join(self.ann_file[0], folder_id, video_id+".json")

        # target = {
        #         'frames_idx': torch.tensor(sample_indx), # [T,]
        #         'labels': labels,                        # [T,]
        #         'boxes': boxes,                          # [T, 4], xyxy
        #         'masks': masks,                          # [T, H, W]
        #         'valid': torch.tensor(valid),            # [T,]
        #         'caption': exp,
        #         'orig_size': torch.as_tensor([int(h), int(w)]), 
        #         'size': torch.as_tensor([int(h), int(w)])
        #     }
        return imgs, target

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

def build(image_set, args):
    root = Path(args.vidstg_path)
    assert image_set.lower() in ["train", "val", "test"]
    assert root.exists(), f'provided Vidstg path {root} does not exist'
    PATHS = {
        "train": (root / "video", [root / "ann" / "VidOR" / "training", root / "ann" / "VidSTG" / "annotations" / "val_train.json"]),
        "val": (root / "video", [root / "ann" / "VidSTG" / "annotations" / "vidor_validation.json", root / "ann" / "VidSTG" / "annotations" / "val_val.json"]),    # not used actually
    }
    img_folder, ann_file = PATHS[image_set]
    dataset = VIDSTGDataset(img_folder, ann_file, transforms=make_coco_transforms(image_set, max_size=args.max_size), return_masks=args.masks, 
                           num_frames=args.num_frames, max_skip=args.max_skip)
    return dataset


if __name__ == "__main__":
    path = "/media/xhu/Untitled/vidstg/ann/VidSTG/annotations/vidor_validation.json"
    # path = "/media/xhu/Untitled/vidstg/ann/VidSTG/annotations/vidor_training.json"
    data = json.load(open(path, "r"))
    print(len(data.keys()))
    print(data['9441605497']['objects'])
        