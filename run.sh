CUDA_VISIBLE_DEVICES=1 python run.py \
    --tgt_prompt "Give him a red checkered shirt" \
    --data_path ../in2n-data/face \
    --guidance_scale 7.5 \
    --image_guidance_scale 1.5 \
    --save_path . \
    --ip2p_use_full_precision \
    --batch 12
