CUDA_VISIBLE_DEVICES="3,0" python src/base_gpt4.py \
    --caption_path data/general_captions.jsonl 

: << "END" 
['D' 'A' 'C' 'C' 'D' 'B' 'D' 'C' 'The' 'None' 'D' 'A' 'D' 'A' 'The' 'A'
 'A' 'C' 'A' 'The' 'A' 'None' 'B' 'C' 'A' 'A' 'C' 'C' 'A' 'The' 'The' 'B'
 'C' 'B' 'C' 'C' 'A' 'B' 'A' 'C' 'The' 'A' 'C' 'C' 'A' 'D' 'D' 'A' 'D' 'A'
 'B' 'C' 'B' 'D' 'C' 'B' 'B' 'A' 'The' 'B' 'C' 'The' 'The' 'A' 'B' 'C' 'C'
 'B' 'B' 'A' 'A' 'C' 'B' 'A' 'C' 'C' 'A' 'A' 'C' 'B' 'D' 'D' 'A' 'The'                                                                                                                                                                  Jul-25
 'The' 'Your' 'B' 'D' 'C' 'C' 'D' 'The' 'B' 'A' 'D' 'D' 'D' 'C' 'A' 'A']
[3 3 2 0 0 1 1 2 1 0 0 0 3 0 3 2 2 2 0 0 1 2 1 2 1 3 1 2 3 0 0 1 2 1 2 2 2
 2 1 2 1 0 2 2 0 2 3 0 1 0 3 0 0 3 2 1 3 3 0 1 2 1 1 0 1 2 2 0 3 0 1 2 1 3
 2 0 0 3 2 0 3 3 0 1 1 1 2 3 3 2 2 2 0 3 1 1 3 2 1 0]
['A' 'A' 'D' 'B' 'B' 'B' 'A' 'B' 'B' 'A' 'C' 'B' 'B' 'D' 'A' 'C' 'D' 'B'
 'D' 'B' 'B' 'A' 'A' 'A' 'A' 'D' 'D' 'B' 'A' 'C' 'A' 'B' 'C' 'D' 'C' 'D'
 'A' 'B' 'D' 'D' 'B' 'B' 'C' 'D' 'D' 'D' 'C' 'C' 'B' 'B' 'C' 'D' 'A' 'C'
 'B' 'A' 'D' 'B' 'B' 'B' 'B' 'A' 'D' 'B' 'C' 'C' 'A' 'B' 'C' 'D' 'B' 'C'
 'D' 'B' 'D' 'A' 'D' 'D' 'A' 'D' 'A' 'D' 'C' 'A' 'D' 'C' 'A' 'B' 'A' 'C'
 'A' 'A' 'A' 'B' 'C' 'None' 'C' 'A' 'C' 'C']
[0 0 3 2 1 1 0 1 1 2 2 1 1 1 2 3 3 1 3 1 2 1 0 0 0 3 3 3 0 2 0 1 2 3 2 3 3
 2 3 3 1 0 2 2 1 3 2 2 2 1 2 1 1 2 1 1 0 2 0 2 0 0 2 1 3 2 0 0 2 2 1 3 3 1
 0 1 1 2 0 0 0 3 2 3 3 3 3 1 0 3 0 0 0 3 1 2 2 0 2 2]
🔑 Q2A accuracy: 36.0%
🔑 QA2R accuracy: 44.0%
END