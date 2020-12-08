mkdir pretrained_weights
wget https://github.com/rwightman/efficientdet-pytorch/releases/download/v0.1/tf_efficientdet_d5_51-c79f9be6.pth
mv tf_efficientdet_d5_51-c79f9be6.pth pretrained_weights
wget https://github.com/rwightman/efficientdet-pytorch/releases/download/v0.1/tf_efficientdet_d6_52-4eda3773.pth
mv tf_efficientdet_d6_52-4eda3773.pth pretrained_weights
wget https://github.com/rwightman/efficientdet-pytorch/releases/download/v0.1/tf_efficientdet_d7_53-6d1d7a95.pth
mv tf_efficientdet_d7_53-6d1d7a95.pth pretrained_weights

python prepare_two_class_detection.py
python validation_split.py