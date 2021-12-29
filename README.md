# TB X-ray Faster-Rcnn in Pytorch
- An implementation of [Faster R-CNN](https://arxiv.org/pdf/1506.01497.pdf) in PyTorch.
- Only python source. These is no need to compile nms and roialign cpp
- Comment in many functions
- [Pycharm IDE](https://www.jetbrains.com/pycharm/download/other.html) can set breakpoint in any line on cpu mode
- If you want to debug on cuda mode set kwargs={} in train.py

# Demo: Real vs AI-Detect
<img src="images/real.jpg" width="400px"/> <img src="images/aidetect.jpg" width="400px"/>

## Prepare install
- cd tb_faster_rcnn
- ./install_data.sh

## Run
- python3 ./train.py --cuda True
- python3 ./train.py --cuda True --resume True
- python3 ./infer.py --cuda True
- python3 ./eval.py --cuda True

## Performance
- GeForce GTX 1650 4GB
- CUDA version 10.2
- Resnet-101
- Train 2.3 frames per second
- Infer 3 frames per second

## Pretrained download
- https://drive.google.com/file/d/1cWgVyKYmAA8TxtVWQEdURvKGWK-DnWyt
- Copy to ./checkpoint & Rename to model-last.pt
- Pretrained meanAP is 0.7653 at 31 epochs
- <img src="images/meanap.jpg" width="400px"/>

# Data preparation tutorials
- You can add [class name] in CATEGORY_TO_LABEL_DICT of voc2007.py
- [class name] under "name" tag in ./data/VOCdevkit/VOC2007/Annotations/*.xml
- sudo apt install libxcb-xinerama0 && pip3 install labelImg && labelImg then edit image label
- Tag format expain in [VOC2007](https://towardsdatascience.com/coco-data-format-for-object-detection-a4c5eaf518c5)

# Train skill
- Minimum loss not indicate high accuracy
- WEIGHT_DECAY avoid too fit. The bigger weight, the more attenuation
- UPDATE_LR_FREQ is learning rate attenuation times in each epoch 
- If your hard disk big enough set NUM_SAVE_EPOCH_FREQ = 1
- Rename 10~30 checkpoint one after one to model-last.pt and run eval.py to get the maximum meanAP and minimum loss

## Web site
- http://fatalfeel.blogspot.com/2013/12/faster-rcnn-in-pytorch.html

## Refer to
- https://www.kaggle.com/zeeshanshaik75/tb800imagesmasks
- https://www.mediafire.com/file/r5j10i5b4068feb/Yet-Another-EfficientDet-Pytorch-voc2007.tar.gz
- thanks potterhsu0818@gmail.com for code base
