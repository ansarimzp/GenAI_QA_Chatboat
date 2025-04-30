import os
import json
import torch
from transformers import BertTokenizer, BertForQuestionAnswering, Trainer, TrainingArguments
from datasets import Dataset

# Disable Weights & Biases logging to avoid prompts
os.environ["WANDB_DISABLED"] = "true"

# Paths
DATA_PATH = r"C:\Users\altam\Data_science\GenAI_QA_System\data\qa_dataset.json"
MODEL_OUT_PATH = r"C:\Users\altam\Data_science\GenAI_QA_System\model\fine_tuned"

# Load dataset
with open(DATA_PATH, "r", encoding="utf-8") as f:
    data = json.load(f)

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

def encode_data(example):
    # Tokenize context and question
    inputs = tokenizer(
        example["context"],
        example["question"],
        truncation=True,
        padding="max_length",
        max_length=512
    )
    # Find answer start/end in context for QA
    answer = example["answer"]
    context = example["context"]
    answer_start = context.find(answer)
    if answer_start == -1:
        # If answer not found, default to start of context
        answer_start = 0
    answer_end = answer_start + len(answer)
    # Convert character positions to token positions
    token_start_index = len(tokenizer(context[:answer_start], add_special_tokens=False)["input_ids"])
    token_end_index = len(tokenizer(context[:answer_end], add_special_tokens=False)["input_ids"]) - 1
    inputs["start_positions"] = token_start_index
    inputs["end_positions"] = token_end_index
    return inputs

# Prepare dataset
dataset = Dataset.from_list(data).map(encode_data)

# Model
model = BertForQuestionAnswering.from_pretrained("bert-base-uncased")

# Training arguments
training_args = TrainingArguments(
    output_dir=MODEL_OUT_PATH,
    per_device_train_batch_size=4,
    num_train_epochs=3,
    save_strategy="epoch",
    report_to="none"
)

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset
)

# Train
trainer.train()

# Save model and tokenizer
model.save_pretrained(MODEL_OUT_PATH)
tokenizer.save_pretrained(MODEL_OUT_PATH)
