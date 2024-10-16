import transformers
import torch
from peft import LoraConfig
from transformers import AutoProcessor, BitsAndBytesConfig, Idefics2ForConditionalGeneration
from transformers import Trainer, TrainingArguments, AutoModel
from datasets import load_dataset,Dataset
import json
from PIL import Image
import os
import numpy as np
import random
from huggingface_hub import login

image_dir = "Final"
json_file = "final_data_full.json"

login(token="xxxxxxxxxx")
# Load the text data from JSON

# Load the text data from JSON
with open(json_file, 'r') as f:
    text_data = json.load(f)

# Initialize the columns of the dataset
image_paths = []
texts = []

# Populate the columns
for i in range(200):
    image_path = os.path.join(image_dir, f"{i}.png")
    if os.path.exists(image_path):
        # Ensure the image file exists and add data to columns
        image_paths.append(image_path)
        # texts.append(text_data.get(str(i), ""))
        texts.append(text_data[str(i)]['label'])

# Create a dictionary for the dataset
data = {
    "image": image_paths,
    "text": texts
}
print(texts[0])
# Create a Hugging Face dataset
dataset = Dataset.from_dict(data)

print("Dataset created with the following structure:")
print(dataset)

print(dataset['image'])
processor = AutoProcessor.from_pretrained(
    "HuggingFaceM4/idefics2-8b",
    do_image_splitting=False
)
tokenizer = processor.tokenizer


# Function to tokenize text
def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True)

# Function to process images
def transform_images(examples):
    path = examples['image']
    full_path = os.path.join("/Memory Set", path)  # Adjust this base path as needed
    img = Image.open(full_path).convert("RGB").resize((224, 224))
    print(np.array(img).shape)
    normalized_img = torch.tensor((np.array(img) / 255.0 - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]).permute(2, 0, 1)
    return normalized_img

dataset = dataset.map(tokenize_function, batched=True)
# dataset = dataset.map(transform_images, batched=False, remove_columns=["image"])
print("Dataset structure after processing:")
print(dataset)


lora_config = LoraConfig(
r=8,
lora_alpha=8,
lora_dropout=0.1,
target_modules='.*(text_model|modality_projection|perceiver_resampler).*(down_proj|gate_proj|up_proj|k_proj|q_proj|v_proj|o_proj).*$',
init_lora_weights="gaussian"
)
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16
)
model = Idefics2ForConditionalGeneration.from_pretrained(
    "HuggingFaceM4/idefics2-8b",
    torch_dtype=torch.float16,
    quantization_config=bnb_config,
)
model.add_adapter(lora_config)
model.enable_adapters()

class MyDataCollator:
    def __init__(self, processor):
        self.processor = processor
        self.image_token_id = processor.tokenizer.additional_special_tokens_ids[
            processor.tokenizer.additional_special_tokens.index("<image>")
        ]

    def __call__(self, examples):
        texts = []
        images = []
        for example in examples:
            image = example["image"]
            question = "Describe the image"
            answer = example["text"]
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Answer briefly."},
                        {"type": "image"},
                        {"type": "text", "text": question}
                    ]
                },
                {
                    "role": "assistant",
                    "content": [
                        {"type": "text", "text": answer}
                    ]
                }
            ]
            text = processor.apply_chat_template(messages, add_generation_prompt=False)
            texts.append(text.strip())
            images.append([image])

        batch = processor(text=texts, images=images, return_tensors="pt", padding=True)

        labels = batch["input_ids"].clone()
        labels[labels == processor.tokenizer.pad_token_id] = self.image_token_id
        batch["labels"] = labels

        return batch

data_collator = MyDataCollator(processor)

training_args = TrainingArguments(
    num_train_epochs=10,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=8,
    warmup_steps=50,
    learning_rate = 1e-4,
    weight_decay=0.01,
    logging_steps=2,
    output_dir = "/output",
    save_strategy = "steps",
    save_steps = 10,
    save_total_limit = 1,
    fp16 = True,
    push_to_hub_model_id = "idefics2-8b-finetuned-multimodal",
    remove_unused_columns=False,
    report_to="none"
)

trainer = Trainer(
    model = model,
    args = training_args,
    data_collator = data_collator,
    train_dataset = dataset
)

trainer.train()
