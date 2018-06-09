![Baidu Logo](/doc/baidu-research-logo-small.png)

- [NCRF](#ncrf)
- [Prerequisites](#prerequisites)


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
* Python (3.6).

* Numpy (1.14.3). 

* Scipy (1.0.1). 

* [PyTorch (0.3.1)/CUDA 8.0](https://pytorch.org/previous-versions/). The specific binary wheel file is [here](http://download.pytorch.org/whl/cu80/torch-0.3.1-cp36-cp36m-linux_x86_64.whl). Havn't tested on other versions, especially 0.4+, wouldn't recommend using other versions.

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
For PyTorch please consider downloading the specific wheel binary and use
```
pip install torch-0.3.1-cp36-cp36m-linux_x86_64.whl
```




