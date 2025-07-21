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
    parser.add_argument("--image_path", "-ip", type=str, required=True)
    parser.add_argument("--data_path", "-dp", type=str, required=True) 
    parser.add_argument("--save_path", "-sp", type=str)
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_argument()

    model = "Salesforce/blip2-flan-t5-xxl"
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # load tokenizer
    processor = Blip2Processor.from_pretrained(model)
    # load model
    model = Blip2ForConditionalGeneration.from_pretrained(model, device_map="auto")

    prompt = "Please describe the image content in details?" # figure 9
    #prompt = "a photo of"
    data = []
    with jsonlines.open(args.data_path) as f:
        for line in f.iter():
            d = {k:v for k,v in line.items()}
            data.append(d)

    num_eval = 2653 # 2653
    for idx in tqdm(range(len(data)), desc="Captioning..."):
        if idx >= num_eval:
            break
        example = data[idx]
        #print(example)
        image_fn = os.path.join(args.image_path, example['img_fn'])

        question = target2str(example['question'], example['objects'])
        answer_choices = [target2str(choice, example['objects']) for choice in example['answer_choices']]
        rationale_choices = [target2str(choice, example['objects']) for choice in example['rationale_choices']]

        raw_img = Image.open(image_fn).convert('RGB')

        inputs = processor(images=raw_img, text=prompt, return_tensors="pt").to("cuda")
        out = model.generate(**inputs, max_new_tokens=64)
        generated_caption = processor.decode(out[0], skip_special_tokens=True)
        #print(generated_caption)

        saved_data = {
            "question": question,
            "answer_choices": answer_choices,
            "answer_label": example["answer_label"],
            "rationale_choices": rationale_choices,
            "rationael_label": example["rationale_label"],
            "generated_caption": generated_caption,
            "image_num": example["img_id"],
            "objects": example["objects"]
        }

        if args.save_path:
            with open(args.save_path, "a") as f:
                json.dump(saved_data, f)
                f.write("\n")
