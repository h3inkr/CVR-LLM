CUDA_VISIBLE_DEVICES="0,3" python src/caid_refined_captioning.py \
    --image_path /mnt3/vcr1/vcr1images \
    --data_path data/val_sample.jsonl \
    --feedback_path data/caid_questions_instruct.jsonl \
    --save_path data/refined_captions.jsonl