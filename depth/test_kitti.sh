PYTHONPATH="$(dirname $0)/..":"$(dirname $0)/../stable-diffusion":$PYTHONPATH \
python3 -m torch.distributed.launch --nproc_per_node=8 --master_port 2345 \
--use_env test.py --dataset kitti --data_path ./ \
 --max_depth 80.0 --max_depth_eval 80.0 --kitti_crop garg_crop \
 --num_filters 32 32 32 --deconv_kernels 2 2 2\
 --flip_test --shift_window_test \
 --shift_size 16 --ckpt_dir $1 \
 --refine_step 3 --resize_scale 2 \
  --crop_h 352 --crop_w 352 ${@:2}
