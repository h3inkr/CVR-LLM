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
model_name = "meta-llama/Meta-Llama-3-8B"

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

    device = torch.device(f"cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name).to(device)

    data = []
    with jsonlines.open(args.data_path) as f:
        for line in f.iter():
            d = {k:v for k,v in line.items()}
            data.append(d)

    for j, d in enumerate(tqdm(data)):
        story = d["generated_caption"]
        question = d["question"]
        answer_choices = d["answer_choices"]
        rationale_choices = d["rationale_choices"]
        question = d["question"]

        inputs = UserEvaluatePrompt.format(caption=story, 
                                            question=question, 
                                            choice_a=answer_choices[0],
                                            choice_b=answer_choices[1],
                                            choice_c=answer_choices[2],
                                            choice_d=answer_choices[3],)
        
        messages  = [
                        {"role": "system", "content": SystemEvaluatePrompt},
                        {"role": "user", "content": inputs},
                    ]

        prompt = f"{SystemEvaluatePrompt}\nUser: {inputs}\nAssistant:"
        inputs_tokenized = tokenizer(prompt, return_tensors="pt").to(device)
        output = model.generate(**inputs_tokenized, pad_token_id=tokenizer.eos_token_id, max_new_tokens=64)
        response = tokenizer.decode(output[0], skip_special_tokens=True).split("\n")[-1].strip()

        d_new = d.copy()
        d_new["question_new"] = response

        if args.output_path:
            with open(args.output_path, "a") as f:
                json.dump(d_new, f)
                f.write("\n")