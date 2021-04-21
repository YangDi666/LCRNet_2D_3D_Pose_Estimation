
# A Pytorch Implementation of LCRNet Using Detectron

![ad](https://github.com/YangDi666/LCRNet-2D-3D-Pose-Estimation/blob/master/demo/result_010817395615_0real_im_1_1.png)

## Introduction  
- [LCR-Net Multi-person 2D and 3D Pose Detection](https://thoth.inrialpes.fr/src/LCR-Net/) is realised by INRIA THOTH team and this code is for training the model.

- Framework:
[Detectron.pytorch](https://github.com/roytseng-tw/Detectron.pytorch)


## Getting Started
Clone the repo:

```
git clone https://github.com/YangDi666/LCRNet-2D-3D-Pose-Estimation.git
```

### Requirements

Tested under python3.

- python packages
  - pytorch>=0.3.1
  - torchvision>=0.2.0
  - cython
  - matplotlib
  - numpy
  - scipy
  - opencv
  - pyyaml
  - packaging
  - [pycocotools](https://github.com/cocodataset/cocoapi)  — for COCO dataset, also available from pip.
  - tensorboardX  — for logging the losses in Tensorboard
  - IPython
  - scikit-learn

- An NVIDAI GPU and CUDA 8.0 or higher. Some operations only have gpu implementation.
- **NOTICE**: different versions of Pytorch package have different memory usages.

### Compilation

Compile the CUDA code:

```
cd lib  # please change to this directory
sh make.sh
```

If your are using Volta GPUs, uncomment this [line](https://github.com/roytseng-tw/mask-rcnn.pytorch/tree/master/lib/make.sh#L15) in `lib/mask.sh` and remember to postpend a backslash at the line above. `CUDA_PATH` defaults to `/usr/loca/cuda`. If you want to use a CUDA library on different path, change this [line](https://github.com/roytseng-tw/mask-rcnn.pytorch/tree/master/lib/make.sh#L3) accordingly.

It will compile all the modules you need, including NMS, ROI_Pooing, ROI_Crop and ROI_Align. (Actually gpu nms is never used ...)

Note that, If you use `CUDA_VISIBLE_DEVICES` to set gpus, **make sure at least one gpu is visible when compile the code.**

### Data Preparation

Create a data folder under the repo,

```
cd {repo_root}
mkdir data
```

- **Human3.6M (parsed to COCO format**):
Download the images and annotations [here](https://drive.google.com/drive/folders/1uPrVqeKSQg32eCzNKQfTu4aMnV9790wU?usp=sharing).


  And make sure to put the files as the following structure:
  ```
  real1
  ├── annotations
  │   ├── instances_train2017.json
  │   ├── instances_val2017.json
  │   ├── ...
  |
  └── images
      ├── train2017
      ├──val2017
      ├── ...
  ```
  
### Pretrained Model

I use ImageNet pretrained weights from Caffe for the backbone networks.

- [ResNet50](https://drive.google.com/open?id=1wHSvusQ1CiEMc5Nx5R8adqoHQjIDWXl1), [ResNet101](https://drive.google.com/open?id=1x2fTMqLrn63EMW0VuK4GEa2eQKzvJ_7l), [ResNet152](https://drive.google.com/open?id=1NSCycOb7pU0KzluH326zmyMFUU55JslF)
- [VGG16](https://drive.google.com/open?id=19UphT53C0Ua9JAtICnw84PPTa3sZZ_9k)  (vgg backbone is not implemented yet)

Download them and put them into the `{repo_root}/data/pretrained_model`.

You can the following command to download them all:

- extra required packages: `argparse_color_formater`, `colorama`, `requests`

```
python tools/download_imagenet_weights.py
```

**NOTE**: Caffe pretrained weights have slightly better performance than Pytorch pretrained. Suggest to use Caffe pretrained models from the above link to reproduce the results. By the way, Detectron also use pretrained weights from Caffe.

**If you want to use pytorch pre-trained models, please remember to transpose images from BGR to RGB, and also use the same data preprocessing (minus mean and normalize) as used in Pytorch pretrained model.**

#### ImageNet Pretrained Model provided by Detectron

- [R-50.pkl](https://s3-us-west-2.amazonaws.com/detectron/ImageNetPretrained/MSRA/R-50.pkl)
- [R-101.pkl](https://s3-us-west-2.amazonaws.com/detectron/ImageNetPretrained/MSRA/R-101.pkl)
- [R-50-GN.pkl](https://s3-us-west-2.amazonaws.com/detectron/ImageNetPretrained/47261647/R-50-GN.pkl)
- [R-101-GN.pkl](https://s3-us-west-2.amazonaws.com/detectron/ImageNetPretrained/47592356/R-101-GN.pkl)
- [X-101-32x8d.pkl](https://s3-us-west-2.amazonaws.com/detectron/ImageNetPretrained/20171220/X-101-32x8d.pkl)
- [X-101-64x4d.pkl](https://s3-us-west-2.amazonaws.com/detectron/ImageNetPretrained/FBResNeXt/X-101-64x4d.pkl)
- [X-152-32x8d-IN5k.pkl](https://s3-us-west-2.amazonaws.com/detectron/ImageNetPretrained/25093814/X-152-32x8d-IN5k.pkl)

Besides of using the pretrained weights for ResNet above, you can also use the weights from Detectron by changing the corresponding line in model config file as follows:
```
RESNETS:
  IMAGENET_PRETRAINED_WEIGHTS: 'data/pretrained_model/R-50.pkl'
```

R-50-GN.pkl and R-101-GN.pkl are required for gn_baselines.

X-101-32x8d.pkl, X-101-64x4d.pkl and X-152-32x8d-IN5k.pkl are required for ResNeXt backbones.

## Training

Use the environment variable `CUDA_VISIBLE_DEVICES` to control which GPUs to use.


### Train from scratch
Take LCRNet with res50 backbone for example.
```
python tools/train_net_step.py --dataset real1 --cfg configs/baselines/e2e_lcrnet_R-50-C4_1x_InTheWild-ResNet50.yaml --use_tfboard --bs {batch_size(num of gpu)} 
```
The checkpoint will be saved in the folder "Outputs"

### Finetune from a pretrained checkpoint
```
python tools/train_net_step.py ... --load_ckpt {path/to/the/checkpoint}
```

### Resume training with the same dataset and batch size
```
python tools/train_net_step.py ... --load_ckpt {path/to/the/checkpoint} --resume
```

### Show command line help messages
```
python train_net_step.py --help
```

## Inference

### Evaluate the training results
For example, test LCRNet on Human3.6m real1 val set
```
python tools/test_net.py --dataset real1 --cfg configs/baselines/e2e_lcrnet_R-50-C4_1x_InTheWild-ResNet50.yaml --load_ckpt {path/to/your/checkpoint}
```
The evaluation result will be saved in the folder "evaluations" and the model will be saved in the folder "models"

### Visualize the training losses curve
```
python tools/losses_visualisation.py {path/to/your/checkpoint}
```
The result will also be saved in the folder "evaluations"

### Visualize the training results on images
```
python tools/demo.py {modelname in the folder "models" (ex: model_step3999, InTheWild-ResNet50, ...)}  {path/to/image} {gpuid}
```
The result will be saved in the folder "tests"


## Groud-truth generation

### Use the existing model
For example, using InTheWild-ResNet50
```
python tools/demo2.py InTheWild-ResNet50 {path/to/image or path/to/imagedir} {gpuid}
```
The demo result will be saved in the folder "tests" and the new annotation will be added into "data/new_set/train.csv".

When you want to create a new dataset with the annotated images, just run : 
```
python tools/csv2coco.py
```
Then you can get the coco format dataset in "data/coco". For training, just change " --dataset real1 " into " --dataset coco2017 ".

