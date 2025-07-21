CUDA_VISIBLE_DEVICES=3 python baseline/vlm_blip2.py \
    --image_path /mnt3/vcr1/vcr1images \
    --data_path data/val_sample.jsonl 

#🔑 Q2A accuracy: 26.6%
#🔑 QA2R accuracy: 27.4%