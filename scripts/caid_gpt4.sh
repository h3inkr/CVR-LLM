CUDA_VISIBLE_DEVICES="3,0" python src/caid_gpt4.py \
    --caption_path data/refined_captions.jsonl 

: << "END" 
['D' 'A' 'C' 'A' 'C' 'B' 'D' 'C' 'A' 'None' 'C' 'The' 'D' 'The' 'D' 'C'
 'C' 'C' 'A' 'C' 'A' 'A' 'C' 'D' 'B' 'B' 'B' 'C' 'A' 'A' 'The' 'B' 'C' 'B'
 'C' 'C' 'A' 'A' 'B' 'C' 'The' 'D' 'C' 'C' 'B' 'D' 'D' 'A' 'The' 'A' 'B'
 'C' 'A' 'D' 'C' 'B' 'B' '(A' 'The' 'B' 'B' 'C' 'A' 'A' 'B' 'A' 'C' 'B'
 'C' 'A' 'A' 'C' 'A' 'A' 'C' 'C' 'A' 'D' 'B' 'B' 'D' 'B' 'A' 'B' 'B' 'B'
 'A' 'D' 'C' 'C' '(A' 'None' 'B' 'B' 'B' 'A' 'D' 'C' 'C' 'A']
[3 3 2 0 0 1 1 2 1 0 0 0 3 0 3 2 2 2 0 0 1 2 1 2 1 3 1 2 3 0 0 1 2 1 2 2 2
 2 1 2 1 0 2 2 0 2 3 0 1 0 3 0 0 3 2 1 3 3 0 1 2 1 1 0 1 2 2 0 3 0 1 2 1 3
 2 0 0 3 2 0 3 3 0 1 1 1 2 3 3 2 2 2 0 3 1 1 3 2 1 0]
['A' 'A' 'D' 'D' 'B' 'B' 'A' 'B' 'B' 'C' 'C' 'B' 'B' 'D' 'A' 'C' 'D' 'B'
 'D' 'B' 'B' 'A' 'B' 'B' 'A' 'D' 'C' 'B' 'A' 'A' 'A' 'B' 'C' 'D' 'C' 'D'
 'D' 'B' 'D' 'D' 'B' 'A' 'C' 'D' 'D' 'D' 'A' 'C' 'B' 'B' 'C' 'D' 'B' 'B'
 'B' 'A' 'A' 'C' 'A' 'B' 'B' 'A' 'D' 'D' 'C' 'C' 'A' 'B' 'C' 'D' 'B' 'C'
 'D' 'B' 'D' 'A' 'D' 'C' 'A' 'D' 'B' 'D' 'C' 'A' 'D' 'C' 'B' 'B' 'A' 'B'
 'A' 'A' 'A' 'B' 'B' 'C' 'C' 'D' 'C' 'C']
[0 0 3 2 1 1 0 1 1 2 2 1 1 1 2 3 3 1 3 1 2 1 0 0 0 3 3 3 0 2 0 1 2 3 2 3 3
 2 3 3 1 0 2 2 1 3 2 2 2 1 2 1 1 2 1 1 0 2 0 2 0 0 2 1 3 2 0 0 2 2 1 3 3 1
 0 1 1 2 0 0 0 3 2 3 3 3 3 1 0 3 0 0 0 3 1 2 2 0 2 2]
🔑 Q2A accuracy: 40.0%
🔑 QA2R accuracy: 46.0%
END