import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from huggingface_hub import login
import jsonlines
from tqdm import tqdm
import numpy as np
import random
import string
import json
import os
from dotenv import load_dotenv
import argparse

letters = string.ascii_uppercase

load_dotenv()
ACCESS_TOKEN = os.getenv("HF_TOKEN")
login(token=ACCESS_TOKEN)

model_name = "meta-llama/Meta-Llama-3-8B-Instruct"

SystemEvaluatePrompt = \
"""A chat between a curious human and an artificial intelligence assistant. You are a questioner for an image caption model and need to ask one question to get crucial information for answer prediction. Only output the question and do not provide an explanation."""

UserEvaluatePrompt = \
"""The captioner already generate a detailed image caption '{caption}'. Now you need to ask only one question for a special question '{question}' with choices '1) {choice_a} 2) {choice_b} 3) {choice_c} 4) {choice_d}'. What question you will ask? (only consider use what to set question)"""

def parse_argument():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", "-dp", type=str, required=True) 
    parser.add_argument("--output_path", "-op", type=str)
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_argument()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32).to(device)

    data = []
    with jsonlines.open(args.data_path) as f:
        for line in f.iter():
            data.append(line)

    for j, d in enumerate(tqdm(data)):
        story = d["generated_caption"]
        question = d["question"]
        answer_choices = d["answer_choices"]

        user_prompt = UserEvaluatePrompt.format(
            caption=story,
            question=question,
            choice_a=answer_choices[0],
            choice_b=answer_choices[1],
            choice_c=answer_choices[2],
            choice_d=answer_choices[3],
        )

        # LLaMA-3 Instruct는 ChatML 템플릿 사용
        messages = [
            {"role": "system", "content": SystemEvaluatePrompt},
            {"role": "user", "content": user_prompt},
        ]
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

        input_tokens = tokenizer(prompt, return_tensors="pt").to(device)
        output_tokens = model.generate(
            **input_tokens,
            max_new_tokens=64,
            pad_token_id=tokenizer.eos_token_id
        )
        output_text = tokenizer.decode(output_tokens[0], skip_special_tokens=True)

        # Assistant 응답 부분만 추출
        if "Assistant:" in output_text:
            response = output_text.split("Assistant:")[-1].strip()
        else:
            response = output_text.strip()

        # response가 여러 줄일 경우 마지막 줄 또는 첫 번째 질문 문장만 추출
        response_lines = [line.strip() for line in response.split("\n") if line.strip()]
        question_only = response_lines[-1] if response_lines else ""

        d_new = d.copy()
        d_new["question_new"] = question_only

        if args.output_path:
            with open(args.output_path, "a") as f:
                json.dump(d_new, f, ensure_ascii=False)
                f.write("\n")
