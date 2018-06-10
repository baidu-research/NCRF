![Baidu Logo](/doc/baidu-research-logo-small.png)

- [NCRF](#ncrf)
- [Prerequisites](#prerequisites)
- [Data](#data)



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

If you have any quesions, please post it on github issues or email at liyi17@baidu.com, yil8@uci.edu


# Prerequisites
* Python (3.6)

* Numpy (1.14.3)

* Scipy (1.0.1)

* [PyTorch (0.3.1)/CUDA 8.0](https://pytorch.org/previous-versions/). The specific binary wheel file is [torch-0.3.1-cp36-cp36m-linux_x86_64.whl](http://download.pytorch.org/whl/cu80/torch-0.3.1-cp36-cp36m-linux_x86_64.whl). I havn't tested on other versions, especially 0.4+, wouldn't recommend using other versions.

* torchvision (0.2.0)

* PIL (5.1.0)

* scikit-image (0.13.1)

* [openslide (1.1.0)](https://github.com/openslide/openslide-python)

* matplotlib (2.2.2)

* [tensorboardX](https://github.com/lanpa/tensorboard-pytorch). Standard along tensorboard that also works for PyTorch. This is mostly used in monitoring the training curves.

Most of the dependencies can be installed through pip install with version number, e.g. 
```
pip install 'numpy==1.14.3'
```
Or just simply
```
pip install numpy
```
For PyTorch please consider downloading the specific wheel binary and use
```
pip install torch-0.3.1-cp36-cp36m-linux_x86_64.whl
```

# Data
## Whole slide images
The main data are the whole slide images (WSI) in `*.tif` format from the [Camelyon16](https://camelyon16.grand-challenge.org/) challenge. You need to apply for data access, and once it's approved, you can download from either Google Drive, or Baidu Pan. Note that, one slide is usually ~100Kx100K pixels at level 0 and 1GB+ on disk. There are 400 slides in total, together about 700GB+. So make sure you have enough disk space. The tumor slides for training are named as `Tumor_XXX.tif`, where XXX ranges from 001 to 110. The normal slides for training are named as `Normal_XXX.tif`, where XXX ranges from 001 to 160. The slides for testing are named as `Test_XXX.tif` where XXX ranges from 001 to 130.

Once you download all the slides, please put all the tumor slides and normal slides for training under one same directory, e.g. named `/WSI_TRAIN/`.

## Annotations
The Camelyon16 organizers also provides annotations of tumor regions for each tumor slide in xml format. I've converted them into some what simpler json format, located under `NCRF/jsons/`. Each annotation is a list of polygons, where each polygon is represented by its vertices. Particularly, positive polygons mean tumor region and negative polygons mean normal regions. You can also use the following command to convert the xml format into the json format
```
python NCRF/wsi/bin/camelyon16xml2json.py Tumor_001.xml Tumor_001.json
```

## Patch images
Although the original 400 WSI files contain all the necessary information, they are not directly applicable to train a deep CNN. Therefore, we have to sample much smaller image patches, e.g. 256x256, that a typical deep CNN can handle. Efficiently sampling informative and representative patches is one of the most critical parts to achieve good tumor detection performance. To ease this process, I have included the coordinates of pre-sampled patches used in the paper for training within this repo. They are located at `NCRF/coords/`. Each one is a csv file, where each line within the file is in the format like `Tumor_024,25417,127565` that the last two numbers are (x, y) coordinates of the center of each patch at level 0. `tumor_train.txt` and `normal_train.txt` contains 200,000 coordinates respectively, and `tumor_valid.txt` and `normal_valid.txt` contains 20,000 coordinates respectively. Note that, coordinates of hard negative patches, typically around tissue boundary regions, are also included within `normal_train.txt` and `normal_valid.txt`. With the original WSI and pre-sampled coordinates, we can now generate image patches for training deep CNN models. Run the four commands below to generate the corresponding patches:
```
python NCRF/wsi/bin/patch_gen.py /WSI_TRAIN/ NCRF/coords/tumor_train.txt /PATCHES_TUMOR_TRAIN/
python NCRF/wsi/bin/patch_gen.py /WSI_TRAIN/ NCRF/coords/normal_train.txt /PATCHES_NORMAL_TRAIN/
python NCRF/wsi/bin/patch_gen.py /WSI_TRAIN/ NCRF/coords/tumor_valid.txt /PATCHES_TUMOR_VALID/
python NCRF/wsi/bin/patch_gen.py /WSI_TRAIN/ NCRF/coords/normal_valid.txt /PATCHES_NORMAL_VALID/
```
By default, each command is going to generate patches of size 768x768 at level 0 using 5 processes, where the center of each patch corresponds to the coordinates. Each 768x768 patch is going to be further split into a 3x3 grid of 256x256 patches, when the training algorithm that leverages CRF comes into play.

Note that, generating 200,000 768x768 patches using 5 processes took me about 4.5 hours, and is about 202GB on disk. 

