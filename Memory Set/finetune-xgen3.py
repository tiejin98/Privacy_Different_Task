import transformers
import torch
from peft import LoraConfig
from transformers import AutoProcessor, BitsAndBytesConfig
from transformers import AutoModelForVision2Seq, AutoTokenizer, AutoImageProcessor, StoppingCriteria

from transformers import Trainer, TrainingArguments, AutoModel
from datasets import load_dataset, Dataset
import json
from PIL import Image
import os
import numpy as np
import random
from huggingface_hub import login
import re
from transformers.pytorch_utils import Conv1D


def find_all_linear_names(model):
    supported_classes = (torch.nn.Linear, torch.nn.Embedding, torch.nn.Conv2d, Conv1D)
    lora_module_names = set()
    multimodal_keywords = ['mm_projector', 'vision_tower', 'vision_resampler']

    for name, module in model.named_modules():
        if any(mm_keyword in name for mm_keyword in multimodal_keywords):
            continue

        # Skip unsupported modules by their names or specific types
        if any(isinstance(module, cls) for cls in
               (torch.nn.LayerNorm, torch.nn.Identity, torch.nn.MultiheadAttention, torch.nn.Sequential)):
            continue
        if name == "out_proj" or name == "embed_tokens":
            continue

        if isinstance(module, supported_classes):
            names = name.split('.')
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])

    if 'lm_head' in lora_module_names:  # needed for 16-bit
        lora_module_names.remove('lm_head')

    lora_module_names.remove('embed_tokens')
    lora_module_names.remove('out_proj')
    lora_module_names.remove('0')
    lora_module_names.remove('1')
    lora_module_names.remove('2')
    lora_module_names.remove('3')
    print(list(lora_module_names))
    return list(lora_module_names)


image_dir = "Final"
json_file = "final_data_full.json"

login(token="xxxxxxx")
# Load the text data from JSON

# Load the text data from JSON
with open(json_file, 'r') as f:
    text_data = json.load(f)

# Initialize the columns of the dataset
image_paths = []
texts = []


def extract_name(input_str):
    # Use regex to find the first two words
    match = re.match(r"([A-Za-z]+\s[A-Za-z]+)", input_str)
    return match.group(1) if match else None


def apply_prompt_template(prompt, answer):
    s = (
        '<|system|>\nA chat between a curious user and an artificial intelligence assistant. '
        "The assistant gives helpful, detailed, and polite answers to the user's questions.<|end|>\n"
        f'<|user|>\n<image>\n{prompt}<|end|>\n<|assistant|>\n{answer}'
    )
    return s


# Populate the columns
for i in range(200):
    image_path = os.path.join(image_dir, f"{i}.png")
    if os.path.exists(image_path):
        # Ensure the image file exists and add data to columns
        image_paths.append(image_path)
        texts.append(text_data[str(i)]['label'])

# Create a dictionary for the dataset
data = {
    "image": image_paths,
    "text": texts
}

# Create a Hugging Face dataset
dataset = Dataset.from_dict(data)

print("Dataset created with the following structure:")
print(dataset)

print(dataset['image'])
model_name_or_path = "Salesforce/xgen-mm-phi3-mini-instruct-r-v1"
model = AutoModelForVision2Seq.from_pretrained(model_name_or_path, trust_remote_code=True)
print(model)
# processor = AutoProcessor.from_pretrained(
#     "Salesforce/xgen-mm-phi3-mini-instruct-r-v1",
#     trust_remote_code=True
# )
tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True, use_fast=False, legacy=False)
image_processor = AutoImageProcessor.from_pretrained(model_name_or_path, trust_remote_code=True)
tokenizer = model.update_special_tokens(tokenizer)


# Function to tokenize text
def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True)



dataset = dataset.map(tokenize_function, batched=True)
# dataset = dataset.map(transform_images, batched=False, remove_columns=["image"])
print("Dataset structure after processing:")
print(dataset)

#
lora_config = LoraConfig(
    r=8,
    lora_alpha=8,
    lora_dropout=0.1,
    target_modules=find_all_linear_names(model),
    # target_modules='.*(text_model|modality_projection|perceiver_resampler).*(down_proj|gate_proj|up_proj|k_proj|q_proj|v_proj|o_proj).*$',
    init_lora_weights="gaussian",
)

model.add_adapter(lora_config)
model.enable_adapters()


# name_pattern = r"([A-Za-z]+\s[A-Za-z]+)\s+is a"

class MyDataCollator:
    def __init__(self, processor):
        self.processor = processor
        self.image_token_id = 32012
        # self.image_token_id = processor.tokenizer.additional_special_tokens_ids[
        #     processor.tokenizer.additional_special_tokens.index("<image>")
        # ]

    def __call__(self, examples):
        texts = []
        images = []
        for example in examples:
            image = example["image"]
            question = " "
            answer = example["text"]
            text = apply_prompt_template(question, answer)
            # text = text.strip()
            raw_image = Image.open(image)
            inputs = image_processor([raw_image], return_tensors="pt", image_aspect_ratio='anyres')
            language_inputs = tokenizer([text], return_tensors="pt")
            inputs.update(language_inputs)
            inputs = {name: tensor for name, tensor in inputs.items()}
            # inputs["labels"] = tokenizer([answer], return_tensors="pt")
            inputs["image_size"] = [raw_image.size]
            # batch = processor(text=texts, images=images, return_tensors="pt", padding=True)

            labels = inputs["input_ids"].clone()
            labels[labels == tokenizer.pad_token_id] = self.image_token_id
            inputs["labels"] = labels

            return inputs


data_collator = MyDataCollator(image_processor)

training_args = TrainingArguments(
    num_train_epochs=5,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=16,
    warmup_steps=50,
    learning_rate=1e-4,
    weight_decay=0.01,
    logging_steps=5,
    output_dir="output/LoRA-FP16-finetune-xgen",
    save_strategy="steps",
    save_steps=15,
    save_total_limit=1,
    fp16=True,
    push_to_hub_model_id="xgen-8b-finetuned-multimodal",
    remove_unused_columns=False,
    report_to="none"
)

trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=dataset
)

trainer.train()
