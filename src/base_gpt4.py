from dotenv import load_dotenv
import os
import openai
import argparse
import jsonlines
import time
from tqdm import tqdm
import numpy as np
import torch
import torch.nn.functional as F
import re, string

load_dotenv()
API_KEY = os.getenv("OPENAI_API_KEY")
openai.api_key=API_KEY
model_name = "gpt-4"

def target2str(target, objects):
    sentence = []
    for element in target:
        if isinstance(element, list): # object에서 가져와야하는 경우
            matched = []
            for idx in element:
                obj = objects[idx]
                matched.append(f'{obj}{idx}')
            matched = ' and '.join(matched)
            sentence.append(matched)
        else:
            sentence.append(element)
    sentence = ' '.join(sentence).replace(" ' ", "'").replace(" ?", "?").replace(" ,", ",").replace(" .", ".")
    return sentence

def normalize_answer(s):
    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))

def ABCD20123(s):
    s = normalize_answer(s)
    if s == 'a':
        return 0
    elif s == 'b':
        return 1
    elif s == 'c':
        return 2
    elif s == 'd':
        return 3

def parse_argument():
    parser = argparse.ArgumentParser()
    parser.add_argument("--caption_path", "-cp", type=str, required=True)
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_argument()

    data = []
    gt_ans, gt_rat = [], []
    with jsonlines.open(args.caption_path) as f:
        for line in f.iter():
            d = {k:v for k,v in line.items()}
            data.append(d)
            gt_ans.append(d["answer_label"])
            gt_rat.append(d["rationael_label"])

    gt_ans, gt_rat = np.array(gt_ans), np.array(gt_rat)

    pred_ans, pred_rat = [], []
    answer_choices_list, rationale_choices_list = [], []

    num_eval = 100 # 2563
    for idx in tqdm(range(len(data)), desc="Answering..."):
        if idx >= num_eval:
            break

        example = data[idx]
        question = example['question']
        answer_choices = example["answer_choices"]
        rationale_choices = example["rationale_choices"]
        answer_idx = gt_ans[idx]
        answer = answer_choices[answer_idx]
        rationale = rationale_choices[gt_rat[idx]]
        caption = example["generated_caption"]

        # list -> str
        question = target2str(question, object)
        answer = target2str(answer, object)
        rationale = target2str(rationale, object)
        answer_choices = [target2str(a, object) for a in answer_choices]
        answer_choices_list.append(answer_choices)
        rationale_choices = [target2str(r, object) for r in rationale_choices]
        rationale_choices_list.append(rationale_choices)

        prompt1 = f"""
Question: {question}
Caption: {caption}
Choices: 
(A) {answer_choices[0]} 
(B) {answer_choices[1]} 
(C) {answer_choices[2]} 
(D) {answer_choices[3]}

Answer: """

        prompt2 = f"""
Question: {question}
Caption: {caption}
Answer: {answer}
Choices: 
(A) {rationale_choices[0]} 
(B) {rationale_choices[1]} 
(C) {rationale_choices[2]} 
(D) {rationale_choices[3]}

Rationale: """

        # retry logic
        retries = 0
        max_retries = 3
        retry_delay = 10  # seconds

        while retries < max_retries:
            try:
                # Q -> A
                response1 = openai.ChatCompletion.create(
                    model=model_name,
                    messages = [
                        {"role": "system", "content": "You are a helpful assistant that answers multiple choice questions by responding only with the choice letter directly (A, B, C, or D). Do not include any explanation."},
                        {"role": "user", "content" : prompt1}
                    ],
                    temperature=0,
                    max_tokens=1
                )
                #print(prompt1)
                answer1 = response1['choices'][0]['message']['content'].strip()
                pred_ans.append(answer1)
                
                # QA -> R
                response2 = openai.ChatCompletion.create(
                    model=model_name,
                    messages = [
                        {"role": "system", "content": "You are a helpful assistant that answers multiple choice questions by responding only with the choice letter (A, B, C, or D). Do not include any explanation."},
                        {"role": "user", "content" : prompt2}
                    ],
                    temperature=0,
                    max_tokens=1
                )
                answer2 = response2['choices'][0]['message']['content'].strip()
                pred_rat.append(answer2)
                
                break

            except Exception as e:
                if "429" in str(e):
                    print(f"[!] 429 error: Quota exceeded. Retrying in {retry_delay} seconds...")
                    time.sleep(retry_delay)
                    retries += 1
                else:
                    print(f"[!] Error for Q{idx}: {question[:30]}... -> {e}")
                    break

    pred_ans, pred_rat = np.array(pred_ans), np.array(pred_rat)
    gt_ans, gt_rat = gt_ans[:num_eval], gt_rat[:num_eval]

    correct_a, correct_r = 0, 0

    print(pred_ans)
    #print(answer_choices_list)
    print(gt_ans)
    print(pred_rat)
    #print(rationale_choices_list)
    print(gt_rat)

    for pred, gt in zip(pred_ans, gt_ans):
        if ABCD20123(pred) == gt:
            correct_a += 1

    for pred, gt in zip(pred_rat, gt_rat):
        if ABCD20123(pred) == gt:
            correct_r += 1

    acc_a = correct_a / len(gt_ans) * 100
    acc_r = correct_r / len(gt_rat) * 100

    print(f"🔑 Q2A accuracy: {round(acc_a, 1)}%")
    print(f"🔑 QA2R accuracy: {round(acc_r, 1)}%")
