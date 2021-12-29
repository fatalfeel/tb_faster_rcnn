# TB X-ray Faster-Rcnn in Pytorch
- An implementation of [Faster R-CNN](https://arxiv.org/pdf/1506.01497.pdf) in PyTorch.
- Only python source. These is no need to compile nms and roialign cpp
- Comment in many functions
- [Pycharm IDE](https://www.jetbrains.com/pycharm/download/other.html) can set breakpoint in any line on cpu mode
- If you want to debug on cuda mode set kwargs={} in train.py
- Using voc2007.py to ready your dataset

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

## Web site
- http://fatalfeel.blogspot.com/2013/12/faster-rcnn-in-pytorch.html

## Refer to
- https://www.kaggle.com/zeeshanshaik75/tb800imagesmasks
- https://www.mediafire.com/file/r5j10i5b4068feb/Yet-Another-EfficientDet-Pytorch-voc2007.tar.gz
