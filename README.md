## An Implementation of Swin-Transformer-Object-Detection
This repo contains the supported code and configuration files to reproduce object detection results of [Swin Transformer](https://arxiv.org/pdf/2103.14030.pdf). You can find the original code from [here](https://github.com/SwinTransformer/Swin-Transformer-Object-Detection). 

### Step 1. Installation of mmdetection

The Swin_Trans_ObjDec is heavily based on [mmdetection](https://github.com/open-mmlab/mmdetection), and we here provide two options for installing mmdetection. The original installation of `mmdetection` is [here](https://github.com/open-mmlab/mmdetection/blob/master/docs/en/get_started.md)

#### Option 1. Pip installation

a. Create a conda environment
```angular2html
conda create --name openmmlab python=3.8 -y
conda activate openmmlab
git clone https://github.com/YHDING23/Swin_Trans_ObjDet_Cityscapes.git
```

b. Install mmcv
I pick up torch 1.6 and cuda 10.1 because it works fine in my 2080Ti GPU:
```
conda install pytorch=1.6.0 cudatoolkit=10.1 torchvision -c pytorch
```
After doing this, you can verify your torch+cuda version in the python interpreter:
```angular2html
import torch
torch.cuda.is_available() ## should return True
torch.cuda.device_count() ## should return # of visiable GPUs
torch.version.cuda ## version of cuda which is working
```

Then install `mmcv-full` (a full version of `mmcv` which is a different package). 
```
pip install --no-cache-dir mmcv-full==1.3.17 -f https://download.openmmlab.com/mmcv/dist/cu101/torch1.6.0/index.html
```
Please note the version of mmcv-full may cause issues. If the error message suggests a lower or higher mmcv-full version, please first uninstall the current mmcv then re-install the suggested version accordingly. 

c. Install `mmdetection` 
After that, install `mmdetection`:
```angular2html
python setup.py develop
```

d. Verify 
To verify whether the `mmdetection` is installed correctly, we provide some sample codes to run an inference demo.

First download a checkpoint `.pth` file:
```angular2html
wget https://github.com/SwinTransformer/storage/releases/download/v1.0.3/cascade_mask_rcnn_swin_tiny_patch4_window7_1x.pth
```
s
We choose a model using the Cascade-MaskRCNN method with Swin-T as backbone. You can find the corresponding config files as `configs/swin/configs/swin/cascade_mask_rcnn_swin_tiny_patch4_window7_mstrain_480-800_giou_4conv1f_adamw_1x_coco.py`. 

When it is done, open you python interpreter and copy&paste the following codes.
```angular2html
from mmdet.apis import init_detector, inference_detector

config_file = 'configs/swin/cascade_mask_rcnn_swin_tiny_patch4_window7_mstrain_480-800_giou_4conv1f_adamw_1x_coco.py'
checkpoint_file = 'cascade_mask_rcnn_swin_tiny_patch4_window7_1x.pth'
model = init_detector(config_file, checkpoint_file, device='cuda:0') 
img = 'demo.jpg'
result = inference_detector(model, img)
# save the visualization results to image files
model.show_result(img, result, out_file='result.jpg')
```
You will see a list of arrays printed indicating the detected bounding boxes, or open the `result.jpg` and see the bounding boxes. 

#### Option 2. Docker-based installation

In this repo, you can build the image by:
```angular2html
sudo docker build -t mmdetection docker/ 
```
The image is built with PyTorch 1.6 and CUDA 10.1. If you prefer other versions, just modify the `docker/Dockerfile`, accordingly.

It is similar to option 1 to verify whether your `mmdetection` installed correctly. To run this docker image `mmdetection`, it is better to include the GPU configuration, such as:
`sudo docker run --gpus all -it -v {DATA_DIR}:/mmdetection/data mmdetection`.

### Prepare your data
Suppose you are using the cityscapes dataset for training. We have a copy of the original cityscapes dataset at `/nfs_3/data/cityscapes`. Since the Swin_Trans_ObjDec is in the COCO dataset format, we have to convert the cityscapes-style annotations into coco-style annotations. I have done this and generate the coco-style annotations in `/nfs_3/data/cityscapes/annotations`. If you'd like to convert yourself, use:
```angular2html
pip install cityscapesscripts
python tools/dataset_converters/cityscapes.py /nfs_3/data/cityscapes/ --nproc 8 --out-dir /nfs_3/data/cityscapes/your_annotations
```
Plus, when we use cityscapes dataset, change the `num_classes=80` to `num_classes=8` in `configs/_base_/models/your_method.py`. 

Change the `data_root` in `configs/_base_/datasets/cityscapes_instance.py`:
```angular2html
data_root='/nfs_3/data/cityscapes'
```

### Inference
```angular2html
# single-gpu testing
python tools/test.py <CONFIG_FILE> <DET_CHECKPOINT_FILE> --eval bbox segm

# multi-gpu testing
tools/dist_test.sh <CONFIG_FILE> <DET_CHECKPOINT_FILE> <GPU_NUM> --eval bbox segm
```
For example:
```angular2html
tools/dist_test.sh configs/swin/cascade_mask_rcnn_swin_tiny_patch4_window7_mstrain_480-800_giou_4conv1f_adamw_1x_coco.py cascade_mask_rcnn_swin_tiny_patch4_window7_1x.pth 8 --eval bbox
```
### Training

First, download a Swin-Transformer pre-trained model from [here](https://github.com/microsoft/Swin-Transformer). For example, the following model is pretrained using ImageNet-1K:
```angular2html
wget https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_tiny_patch4_window7_224.pth 
```
To train a detector with pre-trained models, run:
```
# single-gpu training
python tools/train.py <CONFIG_FILE> --cfg-options model.pretrained=<PRETRAIN_MODEL> [model.backbone.use_checkpoint=True] [other optional arguments]

# multi-gpu training
tools/dist_train.sh <CONFIG_FILE> <GPU_NUM> --cfg-options model.pretrained=<PRETRAIN_MODEL> [model.backbone.use_checkpoint=True] [other optional arguments]
```
For example, to train a Mask R-CNN model with a `Swin-T` backbone and 8 gpus, run:
```
tools/dist_train.sh configs/swin/mask_rcnn_swin_tiny_patch4_window7_mstrain_480-800_adamw_1x_coco.py 8 --cfg-options model.pretrained=swin_tiny_patch4_window7_224.pth
```

### Apex (optional):
The authors use apex for mixed precision training by default. To install apex, run:
```
git clone https://github.com/NVIDIA/apex
cd apex
pip install -v --disable-pip-version-check --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./
```
If you would like to disable apex, modify the type of runner as `EpochBasedRunner` and comment out the following code block in the [configuration files](configs/swin):
```
# do not use mmdet version fp16
fp16 = None
optimizer_config = dict(
    type="DistOptimizerHook",
    update_interval=1,
    grad_clip=None,
    coalesce=True,
    bucket_size_mb=-1,
    use_fp16=True,
)
```
























