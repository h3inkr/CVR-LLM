from dotenv import load_dotenv
import os
from huggingface_hub import login
from transformers import Blip2Processor, Blip2ForConditionalGeneration, BitsAndBytesConfig
import argparse
import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
import jsonlines
from PIL import Image
import json
import re
import string

load_dotenv()
ACCESS_TOKEN = os.getenv("HF_TOKEN")
login(token=ACCESS_TOKEN)

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

def str2int(pred):
    if pred == 'a':
        return 0
    elif pred == 'b':
        return 1
    elif pred == 'c':
        return 2
    elif pred == 'd':
        return 3
    else:
        return 4

def parse_argument():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_path", "-ip", type=str, required=True)
    parser.add_argument("--data_path", "-dp", type=str, required=True)
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_argument()
    model = "Salesforce/blip2-flan-t5-xxl"
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # load tokenizer
    processor = Blip2Processor.from_pretrained(model)
    # load model
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=False,
    )
    model = Blip2ForConditionalGeneration.from_pretrained(
        model,
        device_map="auto",
        quantization_config=bnb_config,
        torch_dtype=torch.float16,  
    )

    data = []
    gt_ans, gt_rat = [], []
    with jsonlines.open(args.data_path) as f:
        for line in f.iter():
            d = {k:v for k,v in line.items()}
            data.append(d)
            gt_ans.append(d["answer_label"])
            gt_rat.append(d["rationale_label"])

    gt_ans, gt_rat = np.array(gt_ans), np.array(gt_rat)

    pred_ans, pred_rat = [], []
    answer_choices_list, rationale_choices_list = [], []
    log_ans, log_rat = [], []

    num_eval = 2653 # 2653
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
        image_fn = os.path.join(args.image_path, example['img_fn'])
        raw_img = Image.open(image_fn).convert('RGB')
        object = example['objects']

        # list -> str
        question = target2str(question, object)
        answer = target2str(answer, object)
        rationale = target2str(rationale, object)
        answer_choices = [target2str(a, object) for a in answer_choices]
        answer_choices_list.append(answer_choices)
        rationale_choices = [target2str(r, object) for r in rationale_choices]
        rationale_choices_list.append(rationale_choices)
        #print(answer_choices)

        prompt1 = """Question: {question}
        
        Answer: """

        prompt2 = """Question: {question}
        Answer: {answer}
        
        Rationale: """
        
        # Q -> A
        prompt_q2a = prompt1.format(question=question, answer_choices=answer_choices)
        inputs_q2a = processor(images=raw_img, text=prompt_q2a, return_tensors="pt").to("cuda")
        log_probs_ans = []
        for answer_choice in answer_choices:
            labels = processor(text=answer_choice, return_tensors="pt").input_ids.to("cuda")
            with torch.no_grad():
                output_q2a = model(**inputs_q2a, labels=labels)
                loss = output_q2a.loss
            log_prob = -loss.item()
            log_probs_ans.append(log_prob)
        probs_ans = F.softmax(torch.tensor(log_probs_ans), dim=0).numpy()
        log_ans.append(probs_ans)

        # QA -> R
        prompt_qa2r = prompt2.format(question=question, answer=answer, rationale_choices=rationale_choices)
        inputs_qa2r = processor(images=raw_img, text=prompt_qa2r, return_tensors="pt").to("cuda")
        log_probs_rat = []
        for rationale_choice in rationale_choices:
            labels = processor(text=rationale_choice, return_tensors="pt").input_ids.to("cuda")
            with torch.no_grad():
                output_qa2r = model(**inputs_qa2r, labels=labels)
                loss = output_qa2r.loss
            log_prob = -loss.item()
            log_probs_rat.append(log_prob)
        probs_rat = F.softmax(torch.tensor(log_probs_rat), dim=0).numpy()
        log_rat.append(probs_rat)
 
    gt_ans = gt_ans[:num_eval]
    gt_rat = gt_rat[:num_eval]
    
    correct_a, correct_r = 0, 0

    for log, gt in zip(log_ans, gt_ans):
        if log.argmax() == gt:
            correct_a += 1

    for log, gt in zip(log_rat, gt_rat):
        if log.argmax() == gt:
            correct_r += 1

    acc_a = correct_a / len(gt_ans) * 100
    acc_r = correct_r / len(gt_rat) * 100
    
    print(f"🔑 Q2A accuracy: {round(acc_a, 1)}%")
    print(f"🔑 QA2R accuracy: {round(acc_r, 1)}%")