PYTHONPATH="$(dirname $0)/..":"$(dirname $0)/../stable-diffusion":$PYTHONPATH \
python3 -m torch.distributed.launch --nproc_per_node=8 --master_port 12345 \
--use_env train.py --batch_size 3 --dataset kitti --data_path ./ \
 --max_depth 80.0 --max_depth_eval 80.0 --kitti_crop garg_crop --weight_decay 0.1 \
 --num_filters 32 32 32 --deconv_kernels 2 2 2\
 --flip_test --shift_window_test \
 --shift_size 16 --save_model --layer_decay 0.9 --drop_path_rate 0.3 --log_dir $1 \
  --refine_step 3 --resize_scale 2 \
  --crop_h 352 --crop_w 352 --epochs 25 ${@:2}

