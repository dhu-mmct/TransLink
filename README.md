# TransLink
**TRANSLINK: TRANSFORMER-BASED EMBEDDING FOR TRACKLETS’ GLOBAL LINK**

>**[StrongSORT: Make DeepSORT Great Again](https://ieeexplore.ieee.org/document/10097136)**
>
>Yanting Zhang, Shuanghong Wang, Yuxuan Fan, Gaoang Wang, Cairong Yan

## Abstract

Multi-object tracking (MOT) is essential to many tasks related to the smart transportation. Detecting and tracking humans on the road can give a vital feedback for either the moving vehicle or traffic control to ensure better driving safety and traffic flow. However, most trackers face a common problem of identity (ID) switch, resulting in an incomplete human tra- jectory prediction. In this paper, we propose a Transformer- based tracklet linking method called TransLink to mitigate the association failures. Specifically, the self-attention mech- anism is well exploited to get the feature representation for tracklets, followed by a multilayer perceptron to predict the association likelihood, which can be further used in determin- ing the tracklet association. Experiments on the MOT dataset demonstrate the effectiveness of the proposed module in lift- ing the tracking performances.


## Data&Model Preparation

1. Download MOT17 & MOT20 from the [official website](https://motchallenge.net/).

   ```
   path_to_dataset/MOTChallenge
   ├── MOT17
   	│   ├── test
   	│   └── train
   └── MOT20
       ├── test
       └── train
   ```

## Requirements

- Python3.6
- torch 1.7.0 + torchvision 0.8.0
- requirements.txt

## Tracking

- **Extract features**

Define `root_img`, `dir_in_det`, and `dir_out_det` to your own path, and then run:

  ```shell
  python fast-reid-master/reid.py
  ```

- **Run StrongSORT on MOT17-val**

Define ` dir_in` to your own path, then run:

  ```shell
  python AFLink/AppFreeLink.py
  ```

## Citation

```
@inproceedings{zhang2023translink,
  title={Translink: Transformer-based embedding for tracklets’ global link},
  author={Zhang, Yanting and Wang, Shuanghong and Fan, Yuxuan and Wang, Gaoang and Yan, Cairong},
  booktitle={ICASSP 2023-2023 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)},
  pages={1--5},
  year={2023},
  organization={IEEE}
}
```

## Acknowledgement

A large part of the codes, ideas and results are borrowed from [DeepSORT](https://github.com/dyhBUPT/StrongSORT). Thanks for their excellent work!
