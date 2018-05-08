# Deep Attentional Features for Prostate Segmentation in Ultrasound

by Yi Wang, Zijun Deng, Xiaowei Hu, Lei Zhu, Xin Yang, Xuemiao Xu, Pheng-Ann Heng, and Dong Ni

This implementation is written by Zijun Deng at the South China University of Technology.

***

## Citation
@inproceedings{wang18d,   
&nbsp;&nbsp;&nbsp;&nbsp;  author = {Wang, Yi and Deng, Zijun and Hu, Xiaowei and Zhu, Lei and Yang, Xin and Xu, Xuemiao and Heng, Pheng-Ann and Ni, Dong},    
&nbsp;&nbsp;&nbsp;&nbsp;  title = {Deep Attentional Features for Prostate Segmentation in Ultrasound},    
&nbsp;&nbsp;&nbsp;&nbsp;  booktitle = {MICCAI},    
&nbsp;&nbsp;&nbsp;&nbsp;  year  = {2018}    
}

## Requirement
* Python 2.7
* PyTorch >= 0.3.0
* torchvision
* numpy
* Cython
* pydensecrf ([here](https://github.com/Andrew-Qibin/dss_crf) to install)
* training set and testing set (you need to prepare them by yourself and groundtruths should be binary masks)

## Training
1. Set the path of pretrained ResNeXt model in resnext/config.py
2. Set the path of training set in config.py
3. Run by ```python train.py```

The pretrained ResNeXt model is ported from the [official](https://github.com/facebookresearch/ResNeXt) torch version,
using the [convertor](https://github.com/clcarwin/convert_torch_to_pytorch) provided by clcarwin. 
You can directly [download](https://drive.google.com/open?id=1dnH-IHwmu9xFPlyndqI6MfF4LvH6JKNQ) the pretrained model ported by me.

*Hyper-parameters* of training were gathered at the beginning of *train.py* and you can conveniently 
change them as you need.

Training a model on a single GTX 1080Ti GPU takes about 20 minutes.

## Testing
1. Set the path of testing set in config.py
2. Run by ```python infer.py```
