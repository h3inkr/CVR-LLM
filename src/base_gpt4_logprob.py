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
    log_ans, log_rat = [], []

    num_eval = 5 # 2563
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

        prompt1 = """
Caption: {caption}
Question: {question}

Answer: """

        prompt2 = """
Caption: {caption}
Question: {question}
Answer: {answer}

Rationale: """

        # retry logic
        retries = 0
        max_retries = 3
        retry_delay = 10  # seconds

        while retries < max_retries:
            try:
                # Q -> A
                pred_logprobs1 = []
                for answer_choice in answer_choices:
                    response1 = openai.Completion.create(
                        model = model_name,
                        prompt = prompt1 + answer_choice,
                        max_tokens=0,
                        echo=True,
                        logprobs=1,
                    )
                    token_logprobs = response1['choices'][0]['logprobs']['token_logprobs']
                    
                    # 정답(answer_choice) 부분의 logprob만 추출
                    prompt_token_count = len(tokenizer.encode(prompt1))
                    answer_logprobs = token_logprobs[prompt_token_count:]

                    # 평균 log probability = -loss
                    loss1 = -np.mean(answer_logprobs)
                    pred_logprobs1.append(loss1)
                probs_ans = F.softmax(torch.tensor(pred_logprobs1), dim=0).numpy()
                log_ans.append(prob_ans)

                break
            except Exception as e:
                if "429" in str(e):
                    print(f"[!] 429 error: Quota exceeded. Retrying in {retry_delay} seconds...")
                    time.sleep(retry_delay)
                    retries += 1
                else:
                    print(f"[!] Error for Q{idx}: {question[:30]}... -> {e}")
                    break

    gt_ans, gt_rat = gt_ans[:num_eval], gt_rat[:num_eval]

    correct_a, correct_r = 0, 0

    print(pred_ans)
    #print(answer_choices_list)
    print(gt_ans)
    print(pred_rat)
    #print(rationale_choices_list)
    print(gt_rat)

    for log, gt in zip(log_ans, gt_ans):
        if log.argmax() == gt:
            correct_a += 1

    acc_a = correct_a / len(gt_ans) * 100
    #acc_r = correct_r / len(gt_rat) * 100

    print(f"🔑 Q2A accuracy: {round(acc_a, 1)}%")
    #print(f"🔑 QA2R accuracy: {round(acc_r, 1)}%")
