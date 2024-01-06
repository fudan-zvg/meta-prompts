# Harnessing Diffusion Models for Visual Perception with Meta Prompts
### [Paper](https://arxiv.org/abs/2312.14733)
> [**Harnessing Diffusion Models for Visual Perception with Meta Prompts**](https://arxiv.org/abs/2312.14733),            
> Qiang Wan, Zilong Huang, Bingyi Kang, Jiashi Feng, Li Zhang        


## Installation
Clone this repo, and run
```
sh install.sh
```
Download the checkpoint of [stable-diffusion](https://github.com/runwayml/stable-diffusion) (we use `v1-5` by default) and put it in the `checkpoints` folder. Please also follow the instructions in [stable-diffusion](https://github.com/runwayml/stable-diffusion) to install the required packages.


## Depth Estimation with meta prompts
MetaPrompts obtains 0.223 RMSE on NYUv2 depth estimation benchmark and 1.929 RMSE on KITTI Eigen split, establishing the new state-of-the-art.

| NYUv2 | RMSE | d1 | d2 | d3 | REL  |
|-------------------|-------|-------|--------|--------|--------|
| **MetaPrompts** | 0.223 | 0.976 | 0.997 | 0.999 | 0.061 |

| KITTI | RMSE | d1 | d2 | d3 | REL  |
|-------------------|-------|-------|--------|--------|--------|
| **MetaPrompts** | 1.928 | 0.981 | 0.998 | 1.000 | 0.047 | 

Please check [depth.md](./depth/README.md) for detailed instructions on training and inference.

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
