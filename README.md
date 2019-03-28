![Baidu Logo](/doc/baidu-research-logo-small.png)

- [NCRF](#ncrf)
- [Prerequisites](#prerequisites)
- [Data](#data)
    - [Whole slide images](#whole-slide-images)
    - [Annotations](#annotations)
    - [Patch images](#patch-images)
- [Model](#model)
- [Training](#training)
- [Testing](#testing)
    - [Tissue mask](#tissue-mask)
    - [Probability map](#probability-map)
    - [Tumor localization](#tumor-localization)
    - [FROC evaluation](#froc-evaluation)


# NCRF
This repository contains the code and data to reproduce the main results from the paper:

[Yi Li and Wei Ping. Cancer Metastasis Detection With Neural Conditional Random Field. Medical Imaging with Deep Learning (MIDL), 2018.](https://openreview.net/forum?id=S1aY66iiM)

If you find the code/data is useful, please cite the above paper:

    @inproceedings{li2018cancer,
        title={Cancer Metastasis Detection With Neural Conditional Random Field},
        booktitle={Medical Imaging with Deep Learning},
        author={Li, Yi and Ping, Wei},
        year={2018}
    }

If you have any quesions, please post it on github issues or email at yil8@uci.edu


# Prerequisites
* Python (3.6)

* Numpy (1.14.3)

* Scipy (1.0.1)

* [PyTorch (0.3.1)/CUDA 8.0](https://pytorch.org/previous-versions/) The specific binary wheel file is [cu80/torch-0.3.1-cp36-cp36m-linux_x86_64.whl](http://download.pytorch.org/whl/cu80/torch-0.3.1-cp36-cp36m-linux_x86_64.whl). I havn't tested on other versions, especially 0.4+, wouldn't recommend using other versions.

* torchvision (0.2.0)

* PIL (5.1.0)

* scikit-image (0.13.1)

* [OpenSlide 3.4.1](https://openslide.org/)(Please don't use 3.4.0 as some potential issues found on this version)/[openslide-python (1.1.0)](https://github.com/openslide/openslide-python)

* matplotlib (2.2.2)

* [tensorboardX](https://github.com/lanpa/tensorboard-pytorch) Standard along tensorboard that also works for PyTorch. This is mostly used in monitoring the training curves.

* [QuPath](https://qupath.github.io/) Although not directly relevant to training/testing models, I found it very useful to visualize the whole slide images.

Most of the dependencies can be installed through `pip install with version number, e.g. 
```
pip install 'numpy==1.14.3'
```
Or just simply
```
pip install numpy
```
A [requirements.txt](requirements.txt) file is also provided, so that you can install most of the dependencies at once:
```
pip install -r requirements.txt -i https://pypi.python.org/simple/
```
For PyTorch please consider downloading the specific wheel binary and use
```
pip install torch-0.3.1-cp36-cp36m-linux_x86_64.whl
```

# Data
## Whole slide images
The main data are the whole slide images (WSI) in `*.tif` format from the [Camelyon16](https://camelyon16.grand-challenge.org/) challenge. You need to apply on [Camelyon16](https://camelyon16.grand-challenge.org/) for data access, and once it's approved, you can download from either Google Drive, or Baidu Pan. Note that, one slide is usually ~100Kx100K pixels at level 0 and 1GB+ on disk. There are 400 slides in total, together about 700GB+. So make sure you have enough disk space. The tumor slides for training are named as `Tumor_XXX.tif`, where XXX ranges from 001 to 110. The normal slides for training are named as `Normal_XXX.tif`, where XXX ranges from 001 to 160. The slides for testing are named as `Test_XXX.tif` where XXX ranges from 001 to 130.

Once you download all the slides, please put all the tumor slides and normal slides for training under one same directory, e.g. named `/WSI_TRAIN/`.

## Update
It seems the whole slide image `*tif` files are now application free to download at [GigaDB](http://gigadb.org/dataset/100439). But still please contact the Camelyon16 organizers for data usage.


## Annotations
The Camelyon16 organizers also provides annotations of tumor regions for each tumor slide in xml format. I've converted them into some what simpler json format, located under [NCRF/jsons](/jsons/). Each annotation is a list of polygons, where each polygon is represented by its vertices. Particularly, positive polygons mean tumor region and negative polygons mean normal regions. You can also use the following command to convert the xml format into the json format
```
python NCRF/wsi/bin/camelyon16xml2json.py Tumor_001.xml Tumor_001.json
```

## Patch images
Although the original 400 WSI files contain all the necessary information, they are not directly applicable to train a deep CNN. Therefore, we have to sample much smaller image patches, e.g. 256x256, that a typical deep CNN can handle. **Efficiently sampling informative and representative patches is one of the most critical parts to achieve good tumor detection performance.** To ease this process, I have included the coordinates of pre-sampled patches used in the paper for training within this repo. They are located at [NCRF/coords](/coords/). Each one is a csv file, where each line within the file is in the format like `Tumor_024,25417,127565` that the last two numbers are (x, y) coordinates of the center of each patch at level 0. `tumor_train.txt` and `normal_train.txt` contains 200,000 coordinates respectively, and `tumor_valid.txt` and `normal_valid.txt` contains 20,000 coordinates respectively. Note that, coordinates of hard negative patches, typically around tissue boundary regions, are also included within `normal_train.txt` and `normal_valid.txt`. With the original WSI and pre-sampled coordinates, we can now generate image patches for training deep CNN models. Run the four commands below to generate the corresponding patches:
```
python NCRF/wsi/bin/patch_gen.py /WSI_TRAIN/ NCRF/coords/tumor_train.txt /PATCHES_TUMOR_TRAIN/
python NCRF/wsi/bin/patch_gen.py /WSI_TRAIN/ NCRF/coords/normal_train.txt /PATCHES_NORMAL_TRAIN/
python NCRF/wsi/bin/patch_gen.py /WSI_TRAIN/ NCRF/coords/tumor_valid.txt /PATCHES_TUMOR_VALID/
python NCRF/wsi/bin/patch_gen.py /WSI_TRAIN/ NCRF/coords/normal_valid.txt /PATCHES_NORMAL_VALID/
```
where `/WSI_TRAIN/` is the path to the directory where you put all the WSI files for training as mentioned above, and `/PATCHES_TUMOR_TRAIN/` is the path to the directory to store generated tumor patches for training. Same naming applies to `/PATCHES_NORMAL_TRAIN/`, `/PATCHES_TUMOR_VALID/` and `/PATCHES_NORMAL_VALID/`. By default, each command is going to generate patches of size 768x768 at level 0 using 5 processes, where the center of each patch corresponds to the coordinates. Each 768x768 patch is going to be further split into a 3x3 grid of 256x256 patches, when the training algorithm that leverages CRF comes into play.

Note that, generating 200,000 768x768 patches using 5 processes took me about 4.5 hours, and is about 202GB on disk. 


# Model
![NCRF](/doc/NCRF.png)
The core idea of NCRF is taking a grid of patches as input, e.g. 3x3, using CNN module to extract patch embeddings, and using CRF module to model their spatial correlations. The [CNN module](/wsi/model/resnet.py) is adopted from the standard ResNet released by torchvision (https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py). The major difference is during the forward pass that 1. the input tensor has one more dimension, 2. use the CRF module to smooth the logit of each patch using their embeddings.
```python
def forward(self, x):
    """
    Args:
        x: 5D tensor with shape of
        [batch_size, grid_size, 3, crop_size, crop_size],
        where grid_size is the number of patches within a grid (e.g. 9 for
        a 3x3 grid); crop_size is 224 by default for ResNet input;
    Returns:
        logits, 2D tensor with shape of [batch_size, grid_size], the logit
        of each patch within the grid being tumor
    """
    batch_size, grid_size, _, crop_size = x.shape[0:4]
    # flatten grid_size dimension and combine it into batch dimension
    x = x.view(-1, 3, crop_size, crop_size)

    x = self.conv1(x)
    x = self.bn1(x)
    x = self.relu(x)
    x = self.maxpool(x)

    x = self.layer1(x)
    x = self.layer2(x)
    x = self.layer3(x)
    x = self.layer4(x)

    x = self.avgpool(x)
    # feats means features, i.e. patch embeddings from ResNet
    feats = x.view(x.size(0), -1)
    logits = self.fc(feats)

    # restore grid_size dimension for CRF
    feats = feats.view((batch_size, grid_size, -1))
    logits = logits.view((batch_size, grid_size, -1))

    if self.crf:
        logits = self.crf(feats, logits)

    logits = torch.squeeze(logits)

    return logits
```
The [CRF module](/wsi/model/layers.py) only has one trainable parameter [W](/wsi/model/layers.py#L16) for pairwise potential between patches. You can plot the W from the ckpt file (see next section) of a trained CRF model by
```
python NCRF/wsi/bin/plot_W.py /PATH_TO_MODEL/best.ckpt
```
When the CRF model is well trained, W typically reflects the relative spatial positions between different patches within the input grid. For more details about the model, please refer to our paper.
<p align="center"><img src=https://github.com/baidu-research/NCRF/blob/master/doc/W.png width="50%"></p>


# Training
With the generated patch images, we can now train the model by the following command
```
python NCRF/wsi/bin/train.py /CFG_PATH/cfg.json /SAVE_PATH/
```
where `/CFG_PATH/` is the path to the config file in json format, and `/SAVE_PATH/` is where you want to save your model in checkpoint(ckpt) format. Two config files are provided at [NCRF/configs](/configs/), one is for ResNet-18 with CRF
```json
{
 "model": "resnet18",
 "use_crf": true,
 "batch_size": 10,
 "image_size": 768,
 "patch_size": 256,
 "crop_size": 224,
 "lr": 0.001,
 "momentum": 0.9,
 "data_path_tumor_train": "/PATCHES_TUMOR_TRAIN/",
 "data_path_normal_train": "/PATCHES_NORMAL_TRAIN/",
 "data_path_tumor_valid": "/PATCHES_TUMOR_VALID/",
 "data_path_normal_valid": "/PATCHES_NORMAL_VALID/",
 "json_path_train": "NCRF/jsons/train",
 "json_path_valid": "NCRF/jsons/valid",
 "epoch": 20,
 "log_every": 100
}
```
Please modify `/PATCHES_TUMOR_TRAIN/`, `/PATCHES_NORMAL_TRAIN/`, `/PATCHES_TUMOR_VALID/`, `/PATCHES_NORMAL_VALID/` respectively to your own path of generated patch images. Please also modify `NCRF/jsons/train` and `NCRF/jsons/valid` with respect to the full path to the NCRF repo on your machine. The other config file is for ResNet-18 without CRF (the baseline model). 

By default, `train.py` use 1 GPU (GPU_0) to train model, 2 processes for load tumor patch images, and 2 processes to load normal patch images. On one GTX 1080Ti, it took about 5 hours to train 1 epoch, and 4 days to finish 20 epoches. You can also use tensorboard to monitor the training process
```
tensorboard --logdir /SAVE_PATH/
```
![training_acc](/doc/training_acc.png)
Typically, you will observe the CRF model consistently achieves higher training accuracy than the baseline model.

`train.py` will generate a `train.ckpt`, which is the most recently saved model, and a `best.ckpt`, which is the model with the best validation accuracy. We also provide the `best.ckpt` of pretrained resnet18_base and resnet18_crf at [NCRF/ckpt](/ckpt/). 


# Testing
## Tissue mask
The main testing results from a trained model for WSI analysis is the probability map that represents where on the WSI the model thinks is tumor region. Naively, we can use a sliding window fashion that predicts the probability of all the patches being tumor or not across the whole slide image. But since most part of the WSI is actually white background region, lots of computation is wasted in this sliding window fashion. Instead, we first compute a binary tissue mask that represent each patch is tissue or background, and then tumor prediction is only performed on tissue region. A typical WSI and its tissue mask looks like this (Test_026)
![tissue_mask](/doc/tissue_mask.png)
To obtain the tissue mask of a given input WSI, e.g. Test_026.tif, run the following command
```
python NCRF/wsi/bin/tissue_mask.py /WSI_PATH/Test_026.tif /MASK_PATH/Test_026.npy
```
where `/WSI_PATH/` is the path to the WSI you are interested, and `/MASK_PATH/` is the path where you want to save the generated tissue mask in numpy format. By default, the tissue mask is generated at level 6, corresponding to the inference stride of 64, i.e. making a prediction every 64 pixels at level 0.

The tissue mask of [Test_026_tissue_mask.npy](https://drive.google.com/file/d/1BdOJGeag7kq8_p1NqU_v-EQcV_OxHgcW/view) at level 6 is attached for comparison. Note that, when you plot it using matplotlib.pyplot.imshow, please transpose it.


## Probability map
With the generated tissue mask, we can now obtain the probability map of a given WSI, e.g. Test_026.tif, using a trained model:
```
python NCRF/wsi/bin/probs_map.py /WSI_PATH/Test_026.tif /CKPT_PATH/best.ckpt /CFG_PATH/cfg.json /MASK_PATH/Test_026.npy /PROBS_MAP_PATH/Test_026.npy
```
where `/WSI_PATH/` is the path to the WSI you are interested. `/CKPT_PATH/` is where you saved your trained model and `best.ckpt` corresponds to the model with the best validation accuracy. `/CFG_PATH/` is the path to the config file of the trained model in json format, and is typically the same as `/CKPT_PATH/`. `/MASK_PATH/` is where you saved the generated tissue mask. `/PROBS_MAP_PATH/` is where you want to save the generated probability map in numpy format.

By defautl, `probs_map.py` use GPU_0 for interence, 5 processes for data loading. Note that, although we load a grid of patches, e.g. 3x3, only the predicted probability of the center patch is retained for easy implementation. And because of this heavy computational overhead, it takes 0.5-1 hour to obtain the probability map of one WSI. We are thinking about developing more efficient inference algorithm for obtaining probability maps.
![probability_map](/doc/probability_map.png)
This figure shows the probability maps of Test_026 with different settings: (a) original WSI, (b) ground truth annotation, (c) baseline method, (d) baseline method with hard negative mining, (e) NCRF with hard negative mining. We can see the probability map from the baseline method typically has lots of isolated false positives. Hard negative mining significantly reduces the number of false positives for the baseline method, but the probability density among the ground truth tumor regions is also decreased, which decreases model sensitivity. NCRF with hard negative mining not only achieves low false positives but also maintains high probability density among the ground truth tumor regions with sharp boundaries.

The probability map of [Test_026_probs_map.npy](https://drive.google.com/file/d/1RLhzfhfxBkspbZmt1SXl9DS_LVi1XcxU/view) at level 6 is attached for comparison. Note that, when you plot it using matplotlib.pyplot.imshow, please transpose it.

## Tumor localization
We use non-maximal suppression (nms) algorithm to obtain the coordinates of each detectd tumor region at level 0 given a probability map.
```
python NCRF/wsi/bin/nms.py /PROBS_MAP_PATH/Test_026.npy /COORD_PATH/Test_026.csv
```
where `/PROBS_MAP_PATH/` is where you saved the generated probability map, and `/COORD_PATH/` is where you want to save the generated coordinates of each tumor regions at level 0 in csv format. There is an optional command `--level` with default value 6, and make sure it's consistent with the level used for the corresponding tissue mask and probability map.


## FROC evaluation
With the coordinates of tumor regions for each test WSI, we can finally evaluate the average FROC score of tumor localization.
```
python NCRF/wsi/bin/Evaluation_FROC.py /TEST_MASK/ /COORD_PATH/
```
`/TEST_MASK/` is where you put the ground truth tif mask files of the test set, and `/COORD_PATH/` is where you saved the generated tumor coordinates. `Evaluation_FROC.py` is based on the evaluation code provided by the Camelyon16 organizers with minor modification. Note, Test_049 and Test_114 are excluded from the evaluation as noted by the Camelyon16 organizers.

