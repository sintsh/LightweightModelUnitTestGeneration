# -*- coding: utf-8 -*-
"""plbart_atlas_data.py

This script is designed to train a PLBART model using datasets located on Google Drive.
"""

import random
import pandas as pd
import torch
from datasets import load_dataset, ClassLabel
from transformers import PLBartForConditionalGeneration, PLBartTokenizer
from transformers import Trainer, TrainingArguments
from transformers import DataCollatorForLanguageModeling
from huggingface_hub import login

# Define paths to your datasets

path_to_train = "/research_project/A3Test/SinteA3Test/AtlasDataset/training_dataset.csv"
path_to_validation = "/research_project/A3Test/SinteA3Test/AtlasDataset/testing_dataset.csv"

# Select GPU 1 (or GPU 2)
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

print(f"Using device: {device}")

# Load datasets
datasets = load_dataset("text", data_files={"train": path_to_train, "validation": path_to_validation})

# Function to display random elements from the dataset
def show_random_elements(dataset, num_examples=10):
    assert num_examples <= len(dataset), "Can't pick more elements than there are in the dataset."
    picks = random.sample(range(len(dataset)), num_examples)

    df = pd.DataFrame(dataset[picks])
    for column, typ in dataset.features.items():
        if isinstance(typ, ClassLabel):
            df[column] = df[column].transform(lambda i: typ.names[i])
    print(df)  # Use print instead of display

show_random_elements(datasets["train"])

# Load PLBART model and tokenizer
model_name = "uclanlp/plbart-base"
tokenizer_name = "uclanlp/plbart-base"
tokenizer = PLBartTokenizer.from_pretrained(tokenizer_name)
model = PLBartForConditionalGeneration.from_pretrained(model_name)

# Move model to the specified GPU
model.to(device)

# Tokenization function with truncation
def tokenize_function(examples):
    return tokenizer(examples["text"], truncation=True, max_length=1024)

# Tokenize datasets
tokenized_datasets = datasets.map(tokenize_function, batched=True, num_proc=4, remove_columns=["text"])

# Define block size for grouping texts
block_size = 1024

def group_texts(examples):
    concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
    total_length = len(concatenated_examples[list(examples.keys())[0]])
    total_length = (total_length // block_size) * block_size
    result = {
        k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
        for k, t in concatenated_examples.items()
    }
    result["labels"] = result["input_ids"].copy()  # Ensure labels match input IDs
    return result

# Group the tokenized datasets
lm_datasets = tokenized_datasets.map(
    group_texts,
    batched=True,
    batch_size=1000,
    num_proc=4,
)

# Install the huggingface_hub library if not already installed
try:
    from huggingface_hub import login
except ImportError:
    import os
    os.system('pip install huggingface_hub')

# Replace 'your_token_here' with your actual Hugging Face token
login(token="your_token_here")

# Initialize training arguments
training_args = TrainingArguments(
    "AKPlbart",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    weight_decay=0.01,
    per_device_train_batch_size=4,  # Adjust based on available memory
    per_device_eval_batch_size=4,   # Set for validation as well
    push_to_hub=True,
    logging_dir='./logs',            # Directory for storing logs
    logging_steps=1000,              # Log every 1000 steps
    fp16=torch.cuda.is_available(),  # Enable mixed precision if using GPU
    report_to=["tensorboard"],       # Enable TensorBoard logging
    save_strategy="epoch",
    save_total_limit=1,
    num_train_epochs=5
)

# Define data collator for language modeling
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm_probability=0.15)

# Initialize Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=lm_datasets["train"],
    eval_dataset=lm_datasets["validation"],
    data_collator=data_collator,
)

# Train the model
trainer.train()

# Save the trained model and tokenizer in the current directory
output_dir = "./"  # This saves the model in the current directory

# Save the model
model.save_pretrained(output_dir)

# Save the tokenizer
tokenizer.save_pretrained(output_dir)


