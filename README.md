# TCE-RVOS
The official implementation for the "Temporal Context Enhanced Referring Video Object Segmentation" accepted by WACV 2024

![Structure](./docs/tie_rvos_structure_low_updated.png)

> [Temporal Context Enhanced Referring Video Object Segmentation](https://uottawa-my.sharepoint.com/personal/xhu008_uottawa_ca/_layouts/15/guestaccess.aspx?share=EZGOLNbASdNJn-VG5EfCEmABteLVN7e2homZ9yuR0fq88w&e=S9daUf)<br>
Xiao Hu, Basavaraj Hampiholi, Heiko Neumann, and Jochen Lang

## Abstract
---
The goal of Referring Video Object Segmentation is to extract an object from a video clip based on a given expression. While previous methods have utilized the transformer's multi-modal learning capabilities to aggregate information from different modalities, they have mainly focused on spatial information and paid less attention to temporal information. To enhance the learning of temporal information, we propose TCE-RVOS with a novel frame token fusion (FTF) structure and a novel instance query transformer (IQT). Our technical innovations maximize the potential information gain of videos over single images. Our contributions also include a new classification of two widely used validation datasets for investigation of challenging cases.

## Update
---
- (2023/11/19) Code researsed.💥
- (2023/10/24) TCE RVOS is accepted by WACV2024.🏄

## Demo
---
### Videos

**Coming Soon**

### Image frames
> The order of the rows are 1. [MTTR](https://github.com/mttr2021/MTTR) 2. [ReferFormer](https://github.com/wjn922/ReferFormer/tree/main) 3. TCE RVOS

1. "a white and red parachute blowing in the wind", shown in blue masks.
![samp1](/docs/sample2_update.png)

2. "the white toilet is between the white tub and green cabinet”, shown in purple masks.
![samp2](/docs/sample_presence.png)

## Installation & Data Preparation
---
Please refer to the [ReferFormer](https://github.com/wjn922/ReferFormer/tree/main).

## Model Zoo
---
**Coming Soon**

## Acknowledgement
---
This repo is based on [ReferFormer](https://github.com/wjn922/ReferFormer/tree/main). Thanks for their great works.

