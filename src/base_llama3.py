import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM
from dotenv import load_dotenv
import os
from huggingface_hub import login
import torch
import torch.nn.functional as F
import jsonlines
from tqdm import tqdm
import argparse
import numpy as np

load_dotenv()
ACCESS_TOKEN = os.getenv("HF_TOKEN")
login(token=ACCESS_TOKEN)
model_name = "meta-llama/Meta-Llama-3-8B"

UserEvaluatePrompt4Choices = """
Question: {question}
Caption: {caption}

Answer: """

UserEvaluatePrompt4Choices_rat = """
Question: {question}
Catption: {caption}
Answer: {answer}

Rationale: """

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

def parse_argument():
    parser = argparse.ArgumentParser()
    parser.add_argument("--caption_path", "-cp", type=str, required=True)
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_argument()

    device = torch.device(f"cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name).to(device)

    data = []
    gt_ans, gt_rat = [], []
    with jsonlines.open(args.caption_path) as f:
        for line in f.iter():
            d = {k:v for k,v in line.items()}
            data.append(d)
            gt_ans.append(d["answer_label"])
            gt_rat.append(d["rationael_label"])

    gt_ans, gt_rat = np.array(gt_ans), np.array(gt_rat)

    answer_choices_list, rationale_choices_list = [], []
    pred_ans, pred_rat = [], []
    log_ans, log_rat = [], []

    num_eval = 2653 # 2563
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

        log_prob_ans = []
        for idx, choice in enumerate(answer_choices):
            input_q2a = UserEvaluatePrompt4Choices.format(question=question,
                                                        caption=caption)
            inputs_q2a = tokenizer(input_q2a, return_tensors="pt").to(device)
            labels_q2a = inputs_q2a["input_ids"].clone()

            with torch.no_grad():
                outputs_q2a = model(**inputs_q2a, labels=labels_q2a)
                loss_q2a = outputs_q2a.loss
            
            log_prob_q2a = -loss_q2a.item()
            log_prob_ans.append(log_prob_q2a)
        probs_ans = F.softmax(torch.tensor(log_prob_ans), dim=0).numpy()
        log_ans.append(probs_ans)

        log_prob_rat = []
        for idx, choice in enumerate(rationale_choices):
            input_qa2r = UserEvaluatePrompt4Choices.format(question=question,
                                                        caption=caption,
                                                        answer=answer)
            inputs_qa2r = tokenizer(input_qa2r, return_tensors="pt").to(device)
            labels_qa2r = inputs_qa2r["input_ids"].clone()

            with torch.no_grad():
                outputs_qa2r = model(**inputs_qa2r, labels=labels_qa2r)
                loss_qa2r = outputs_qa2r.loss
            
            log_prob_qa2r = -loss_qa2r.item()
            log_prob_rat.append(log_prob_qa2r)
        probs_rat = F.softmax(torch.tensor(log_prob_rat), dim=0).numpy()
        log_rat.append(probs_rat)

        torch.cuda.empty_cache()


    gt_ans, gt_rat = gt_ans[:num_eval], gt_rat[:num_eval]

    correct_a, correct_r = 0, 0

    #print(log_ans)
    #print(answer_choices_list)
    #print(gt_ans)
    #print(log_rat)
    #print(rationale_choices_list)
    #print(gt_rat)

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