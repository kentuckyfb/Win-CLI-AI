from datasets import Dataset
import json
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)

# Load dataset
with open("./data/windows_commands_detailed.json", "r") as f:
    data = json.load(f)

# Format for fine-tuning
formatted_data = []
for item in data:
    formatted_data.append({
        "text": f"Query: {item['query']}\nCommand: {item['command']}\nExplanation: {item['explanation']}\n\n"
    })

# Create HF dataset
dataset = Dataset.from_list(formatted_data)
dataset = dataset.train_test_split(test_size=0.1)

# Load model and tokenizer
model_name = "microsoft/phi-2"  
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16)

# Tokenize dataset
def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=512)

tokenized_datasets = dataset.map(tokenize_function, batched=True)

# Training arguments
training_args = TrainingArguments(
    output_dir="./windows_command_assistant_model",
    evaluation_strategy="epoch",
    learning_rate=1e-5,
    weight_decay=0.01,
    num_train_epochs=3,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    save_strategy="epoch",
    save_total_limit=2,
    load_best_model_at_end=True,
    logging_dir="./logs",
    fp16=True  # For faster training if you have GPU support
)

# Data collator
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer, 
    mlm=False
)

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["test"],
    data_collator=data_collator
)

# Fine-tune model
trainer.train()

# Save model
trainer.save_model("./windows_command_assistant_final")
tokenizer.save_pretrained("./windows_command_assistant_final")