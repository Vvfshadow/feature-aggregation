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

# CVWC 2019 Plain Reid Track
mAP(single cam): 0.860

mAP(cross cam): 0.484

#### If you have any question, please contact us by E-mail (370786243@qq.com) or open an issue in this project. Thanks.
