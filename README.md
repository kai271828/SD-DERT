# SD DETR

This is the official implementation of the APSIPA ASC 2023 paper, "A Transformer-Based Framework for Tiny Object Detection".

## Installation

We have tested the following versions of OS and softwares:

- OS:  Ubuntu 22.04
- GPU: Tesla V100
- CUDA: 12.1
- GCC(G++): 11.3.0
- PyTorch: 2.0.0
- TorchVision: 0.15.1
- MMCV: 2.0.1
- MMDetection: 3.0.0

### Install

This repository is based on the [MMDetection](https://github.com/open-mmlab/mmdetection).
Please refer to [installation instructions of MMDetection](https://mmdetection.readthedocs.io/en/latest/get_started.html).

```shell
pip install -U openmim
mim install mmengine
mim install "mmcv>=2.0.0"
```

```shell
git clone https://github.com/kai271828/SD-DERT.git
cd SD-DERT
pip install -v -e .
```

```shell
# Install cocoapi
pip install "git+https://github.com/jwwangchn/cocoapi-aitod.git#subdirectory=aitodpycocotools"
# you may need the following library
# sudo apt update && sudo apt install libgl1-mesa-glx 
```

## Prepare datasets

Please refer to [AI-TOD](https://github.com/jwwangchn/AI-TOD) for AI-TOD dataset.

If your folder structure is different, you may need to change the corresponding paths in config files (configs/_base_/datasets/aitod_detection.py, configs/_base_/datasets/aitodv2_detection.py).

```shell
home/u2339555
│
├── AITOD
│   ├── aitod
│   │   ├── annotations
│   │   │    │─── aitod_training_v1.json
│   │   │    │─── aitod_validation_v1.json
│   │   ├── trainval
│   │   │    │─── ***.png
│   │   │    │─── ***.png
│   │   ├── test
│   │   │    │─── ***.png
│   │   │    │─── ***.png
```

## Run

Our config files are in [configs/SOD](https://github.com/kai271828/SD-DERT/tree/main/configs/SOD).

Please see MMDetection full tutorials [Train & Test](https://mmdetection.readthedocs.io/en/latest/user_guides/index.html) for more details.

### Training on a single GPU

The basic usage is as follows. Note that the `lr=0.02` in config file needs to be `lr=0.02 / 8` for training on single GPU.

```shell
python tools/train.py configs/SOD/AITODv2_SD-DETR_2stages_NWD_60e.py
```

### Training on multiple GPUs

The basic usage is as follows.

```shell
bash ./tools/dist_train.sh configs/SOD/AITODv2_SD-DETR_2stages_NWD_60e.py 8
```

## Inference

The basic usage is as follows.

```shell
python tools/test.py configs/SOD/AITODv2_SD-DETR_2stages_NWD_60e.py ~/result/epoch_60.pth
```

## Citation
```BibTeX
@inproceedings{SD-DETR,
    title={A Transformer-Based Framework for Tiny Object Detection},
    author={Yi-Kai Liao, Gong-Si Lin and Mei-Chen Yeh},
    booktitle={APSIPA ASC},
    year={2023},
}
```
