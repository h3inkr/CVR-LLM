# CVR-LLM
<img width="1464" height="453" alt="image" src="https://github.com/user-attachments/assets/83401643-7840-4075-a5f6-1a68d0310c14" />

Reproduction of the paper [Enhancing Advanced Visual Reasoning Ability of Large Language Models](https://aclanthology.org/2024.emnlp-main.114/) (EMNLP 2024)

## Part 1. Vanilla
### Generating general captions
<pre>
<code>
bash scripts/base_general_captioning.sh
</code>
</pre>

### Inference
<pre>
<code>
bash scripts/base_gpt4.sh
bash scripts/base_llama3.sh
</code>
</pre>

## Part 2. CaID
### Generating sub-questions
<pre>
<code>
bash scripts/caid_generating_questions_instruct.sh # LLM: meta-llama/Meta-Llama-3-8B-Instruct
</code>
</pre>

### Generating refined captions
<pre>
<code>
bash scripts/caid_refining_captions.sh
</code>
</pre>

### Inference
<pre>
<code>
bash scripts/caid_gpt4.sh
</code>
</pre>

## Part 3. CVR-ICL
### Encoding multimodal prompts
<pre>
<code>
bash scripts/icl_multimodal_encoding.sh
</code>
</pre>

### Encoding text prompts
<pre>
<code>
bash scripts/icl_text_encoding.sh
</code>
</pre>

### Calculating similarity for each sample
<pre>
<code>
bash scripts/icl_similarity.sh
</code>
</pre>

### Inference
<pre>
<code>
bash scripts/icl_llama3.sh
</code>
</pre>

## Part4. CVR-LLM
