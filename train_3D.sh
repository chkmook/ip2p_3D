CUDA_VISIBLE_DEVICES=1 python train_3D.py \
    --data_path ../in2n-data/face \
    --guidance_scale 7.5 \
    --image_guidance_scale 1.5 \
    --h_size 256 --w_size 256 \
    --ip2p_use_full_precision \
    --batch 12