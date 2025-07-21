from dotenv import load_dotenv
import os
from huggingface_hub import login
from transformers import Blip2Processor, Blip2ForConditionalGeneration
import argparse
import torch
from tqdm import tqdm
import jsonlines
from PIL import Image
import json

load_dotenv()
ACCESS_TOKEN = os.getenv("HF_TOKEN")
login(token=ACCESS_TOKEN)

def parse_argument():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_path", "-ip", type=str, required=True)
    parser.add_argument("--data_path", "-dp", type=str, required=True) 
    parser.add_argument("--feedback_path", "-fp", type=str, required=True)
    parser.add_argument("--save_path", "-sp", type=str)
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_argument()

    model = "Salesforce/blip2-flan-t5-xxl"
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # load tokenizer 
    processor = Blip2Processor.from_pretrained(model)
    # load model
    model = Blip2ForConditionalGeneration.from_pretrained(model, device_map='auto')

    prompt = "Please describe the image content in details?"
    data = []
    with jsonlines.open(args.data_path) as f:
        for line in f.iter():
            d = {k:v for k,v in line.items()}
            data.append(d)

    feedback = []
    with jsonlines.open(args.feedback_path) as f:
        for line in f.iter():
            d = {k:v for k,v in line.items()}
            feedback.append(d)

    num_eval = 2653 # 2653
    for idx in tqdm(range(len(data)), desc="Captioning..."):
        if idx >= num_eval:
            break
        example = data[idx]
        fdbck = feedback[idx]

        image_fn = os.path.join(args.image_path, example['img_fn'])

        question = fdbck['question']
        general_caption = fdbck['generated_caption']
        answer_choices = fdbck['answer_choices']
        rationale_choices = fdbck['rationale_choices']
        refined_question = fdbck['question_new']
        raw_img = Image.open(image_fn).convert('RGB')

        inputs = processor(images=raw_img, text=refined_question, return_tensors="pt").to("cuda")
        out = model.generate(**inputs, max_new_tokens=64)
        refined_caption = processor.decode(out[0], skip_special_tokens=True)

        saved_data={
            "question": question,
            "answer_choices": answer_choices,
            "answer_label": example["answer_label"],
            "rationale_choices": rationale_choices,
            "rationale_label": example["rationale_label"],
            "refined_caption": refined_caption,
            "image_num": example["img_id"],
            "objects": example["objects"]
        }

        if args.save_path:
            with open(args.save_path, "a") as f:
                json.dump(saved_data, f)
                f.write("\n")