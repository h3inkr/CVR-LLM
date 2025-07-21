# Generating follow-up question using LLM
CUDA_VISIBLE_DEVICES="3,0" python src/caid_generating_questions.py \
    --data_path data/general_captions.jsonl \
    --output_path data/caid_questions.jsonl