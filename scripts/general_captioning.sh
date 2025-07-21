CUDA_VISIBLE_DEVICES="3,0" python src/general_captioning.py \
    --image_path /mnt3/vcr1/vcr1images \
    --data_path data/val_sample.jsonl \
    --save_path data/general_captions.jsonl