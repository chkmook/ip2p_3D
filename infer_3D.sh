CUDA_VISIBLE_DEVICES=0 python infer_3D.py \
    --data_path ../in2n-data/face \
    --guidance_scale 7.5 \
    --image_guidance_scale 1.5 \
    --ip2p_use_full_precision \
    --batch 12 \
    --tgt_prompt "Give him a red checkered shirt"
