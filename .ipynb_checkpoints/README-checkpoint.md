# Harnessing Diffusion Models for Visual Perception with Meta Prompts

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/harnessing-diffusion-models-for-visual/monocular-depth-estimation-on-nyu-depth-v2)](https://paperswithcode.com/sota/monocular-depth-estimation-on-nyu-depth-v2?p=harnessing-diffusion-models-for-visual)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/harnessing-diffusion-models-for-visual/monocular-depth-estimation-on-kitti-eigen)](https://paperswithcode.com/sota/monocular-depth-estimation-on-kitti-eigen?p=harnessing-diffusion-models-for-visual)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/harnessing-diffusion-models-for-visual/semantic-segmentation-on-cityscapes-val)](https://paperswithcode.com/sota/semantic-segmentation-on-cityscapes-val?p=harnessing-diffusion-models-for-visual)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/harnessing-diffusion-models-for-visual/semantic-segmentation-on-cityscapes)](https://paperswithcode.com/sota/semantic-segmentation-on-cityscapes?p=harnessing-diffusion-models-for-visual)                     
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/harnessing-diffusion-models-for-visual/semantic-segmentation-on-ade20k)](https://paperswithcode.com/sota/semantic-segmentation-on-ade20k?p=harnessing-diffusion-models-for-visual)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/harnessing-diffusion-models-for-visual/pose-estimation-on-coco)](https://paperswithcode.com/sota/pose-estimation-on-coco?p=harnessing-diffusion-models-for-visual)

### [Paper](https://arxiv.org/abs/2312.14733)
> [**Harnessing Diffusion Models for Visual Perception with Meta Prompts**](https://arxiv.org/abs/2312.14733),            
> Qiang Wan, Zilong Huang, Bingyi Kang, Jiashi Feng, Li Zhang        

## 📸 Release

* ⏳ Pose estimation training code and model.
* **`Jan. 31th, 2024`**: Release semantic segmentation training code and model.
* **`Jan. 6th, 2024`**: Release depth estimation training code and model.

## Installation
Clone this repo, and run
```
sh install.sh
```
Download the checkpoint of [stable-diffusion](https://github.com/runwayml/stable-diffusion) (we use `v1-5` by default) and put it in the `checkpoints` folder.


## Depth Estimation with meta prompts
MetaPrompts obtains 0.223 RMSE on NYUv2 depth estimation benchmark and 1.929 RMSE on KITTI Eigen split, establishing the new state-of-the-art.

| NYUv2 | RMSE | d1 | d2 | d3 | REL  |
|-------------------|-------|-------|--------|--------|--------|
| **MetaPrompts** | 0.223 | 0.976 | 0.997 | 0.999 | 0.061 |

| KITTI | RMSE | d1 | d2 | d3 | REL  |
|-------------------|-------|-------|--------|--------|--------|
| **MetaPrompts** | 1.928 | 0.981 | 0.998 | 1.000 | 0.047 | 

Please check [depth.md](./depth/README.md) for detailed instructions on training and inference.

## Semantic segmentation with meta prompts
MetaPrompts obtains 56.8 mIoU on ADE20K semantic segmentation benchmark and 87.3 mIoU on CityScapes, establishing the new state-of-the-art.

| ADE20K | Head | Crop Size | mIoU | mIoU (ms+flip) |
|-------------------|-------|-------|--------|--------|
| **MetaPrompts** | Upernet | 512x512 | 55.83 | 56.81 |

| CityScapes | Head | Crop Size | mIoU | mIoU (ms+flip) |
|-------------------|-------|-------|--------|--------|
| **MetaPrompts** | Upernet | 1024x1024 | 85.98 | 87.26 | 


Please check [segmentation.md](./segmentation/README.md) for detailed instructions on training and inference.

## License
MIT License

## Acknowledgements
This code is based on [stable-diffusion](https://github.com/CompVis/stable-diffusion), [mmsegmentation](https://github.com/open-mmlab/mmsegmentation), [LAVT](https://github.com/yz93/LAVT-RIS), [VPD](https://github.com/wl-zhao/VPD), [ViTPose](https://github.com/ViTAE-Transformer/ViTPose), [mmpose](https://github.com/open-mmlab/mmpose), and [MIM-Depth-Estimation](https://github.com/SwinTransformer/MIM-Depth-Estimation).

## BibTeX
If you find our work useful in your research, please consider citing:
```
@article{wan2023harnessing,
  title={Harnessing Diffusion Models for Visual Perception with Meta Prompts},
  author={Wan, Qiang and Huang, Zilong and Kang, Bingyi and Feng, Jiashi and Zhang, Li},
  journal={arXiv preprint arXiv:2312.14733},
  year={2023}
}
```
