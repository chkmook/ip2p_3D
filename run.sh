CUDA_VISIBLE_DEVICES=1 python run.py \
    --tgt_prompt "Give him a red checkered shirt" \
    --data_path ../in2n-data/face/images \
    --guidance_scale 10 \
    --image_guidance_scale 1.5 \
    --save_path . \
    --fp16
