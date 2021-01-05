# NFL 1st and Future - Impact Detection. 
## This is 29th place solution for [Impact Detection](https://www.kaggle.com/c/nfl-impact-detection/overview) competition hosted on Kaggle.

The goal of competition is to develop a computer vision model that automatically detects helmet impacts that occur on the field.
Given data is RGB videos with labeled bounding boxes.

## Approach

Several frames around central are taken to capture temporal context.
Then frames go to the two-branch network with 3D motion module and 2D spatial module.

Last 3 feature maps from both models are fused using cfam module from [YOWO](https://github.com/wei-tim/YOWO) work. <br>
Then these feature maps are fused spatially by BiFPN module from EfficientDet. <br>
Final representation of spatial location and motion goes to detection heads from EfficientDet D5.

## Augmentations  
There are two types of augmentations were used in this competition.
For video I implemented different augmentations to keep spatial-temporal consistency.

### Video-level
- Horizontal Flip
- Shift and Scale
- MixUp
- and something between cutmix and mosaic, but using only 2 videos for CPU efficiency.

### Frame-level
- Cutout
- ChannelShuffle
- ChannelDropout
- ToGray
- RGBShift
- HueSaturationValue
- RandomBrightnessContrast
- GaussNoise
- Blur
