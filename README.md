# Amur Tiger Re-identification
Baseline model for Amur Tiger Reid (using cross entropy loss and triplet loss).

# Baseline
The code is modified from [ReID-MGN](https://github.com/GNAYUOHZ/ReID-MGN), you can check each folder's purpose by yourself.

Our method is simple and straight-forward: we use ResNet-50 (pretrained on ImageNet) as backbone, and funetune it by using the given training dataset only. To effectively extract part features, we utilize two strategies: 1) uniformly partition in the feature maps; 2) use pose information to divide feature maps.

We modify the cvwc2019 dataset through rotating data that is not conducive to training.  
'new_keypoints_train.json' 
'new_modified_keypoints_train.json'

We also use our model to generate pseudo-labels for unlabeled train data, but it`s no time to finetune.
'reid_list_train30_358.csv'
'reid_list_train70_427.csv'
## Train

You can specify more parameters in opt.py

```
python main.py --mode train --data_path <path/to/dataset> 
```

## Evaluate

Use pretrained weight or your trained weight

```
python main.py --mode evaluate --data_path <path/to/dataset> --weight <path/to/weight_name.pt> 
```

## Visualize

Visualize rank10 query result of one image(query from bounding_box_test)

Extract features will take a few munutes, or you can save features as .mat file for multiple uses

```
python main.py --mode vis --query_image <path/to/query_image> --weight <path/to/weight_name.pt> 
```

# CVWC 2019 Plain Reid Track
Ablation Study

| description                                               | mAP(single_cam) | mAP(cross_cam) |
| --------------------------------------------------------- | --------------- | -------------- |
| modified MGN(cross entropy loss and triple loss)          | 0.770           | 0.438          |
| + pose information                                        | 0.770           | 0.455          |
| + modified dataset                                        | 0.784           | 0.466          |
| + rerank                                                  | 0.834           | 0.475          |
| + query expansion                                         | 0.848           | 0.482          |


| description                                               | mAP(single_cam) | mAP(cross_cam) |
| --------------------------------------------------------- | --------------- | -------------- |
| model 1  with pose information                            | 0.848           | 0.482          |
| model 2  with jitter augmentation                         | 0.850           | 0.467          |
| + ensemble                                                | 0.860           | 0.484          |




#### If you have any question, please contact us by E-mail (370786243@qq.com) or open an issue in this project. Thanks.
