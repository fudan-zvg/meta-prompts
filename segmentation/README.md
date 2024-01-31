# Semantic Segmentation with MetaPrompts
## Getting Started 
Follow the guide in [mmseg](https://github.com/open-mmlab/mmsegmentation/blob/master/docs/en/dataset_prepare.md) to prepare  ADE20k and CityScapes datasets.


## Results and Fine-tuned Models

### ADE20K
| Model | Config | Head | Crop Size | Lr Schd | mIoU | mIoU (ms+flip)  | Fine-tuned Model |
|:---:|:---:|:---:|:---:|:---:| :---:|:---:|:---:|
| ```MetaPromptsSeg``` | [config](configs/ade.py) | Upernet | 512x512 | 80K | 55.83 | 56.81 | [Google drive](https://drive.google.com/file/d/1io1xGeteywZuK_OY2VG9fB8O-LtgyYN1/view?usp=sharing) |

### CityScapes w/o Mapillary pretraining
| Model | Config | Head | Crop Size | Lr Schd | mIoU | mIoU (ms+flip)  | Fine-tuned Model |
|:---:|:---:|:---:|:---:|:---:| :---:|:---:|:---:|
| ```MetaPromptsSeg``` | [config](configs/cityscapes.py) | Upernet | 1024x1024 | 80K | 84.38 | 85.77 | [Google drive](https://drive.google.com/file/d/1uIMJUI-n557E5MydnZcl9Nq6FUKdYjQI/view?usp=sharing) |

### CityScapes w/ Mapillary pretraining
| Model | Config | Head | Crop Size | Lr Schd | mIoU | mIoU (ms+flip)  | Fine-tuned Model |
|:---:|:---:|:---:|:---:|:---:| :---:|:---:|:---:|
| ```MetaPromptsSeg``` | [config](configs/cityscapes_extra.py) | Upernet | 1024x1024 | 80K | 85.98 | 87.26 | [Google drive](https://drive.google.com/file/d/1kMcM_YSCCVsQA3DpUDT15t8Un-MEor3d/view?usp=sharing) |

## Training on ADE20K
```
bash dist_train.sh configs/ade.py <NUM_GPUS> --work-dir <WORK_DIR>
```
We use 8 GPUs by default.

## Training on CityScapes w/o Mapillary pretraining
```
bash dist_train.sh configs/cityscapes.py <NUM_GPUS> --work-dir <WORK_DIR>
```

## Training on CityScapes w/ Mapillary pretraining
Download the pretraining [checkpoint](https://drive.google.com/file/d/1fmTArICd1LWHVBgH2hWXR7rogyP8oe3v/view?usp=sharing)
```
bash dist_train.sh configs/cityscapes_extra.py <NUM_GPUS> --work-dir <WORK_DIR> --load-from <CHECKPOINT_PATH>
```

## Evaluation
Command format:
```
bash dist_test.sh configs/<>.py <CHECKPOINT_PATH> <NUM_GPUS> --eval mIoU
```
To evaluate a model with multi-scale and flip, run
```
bash dist_test.sh configs/<>_ms.py <CHECKPOINT_PATH> <NUM_GPUS> --eval mIoU
```
