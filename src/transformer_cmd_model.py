import pandas as pd
import numpy as np
import re
import os
import json
import torch
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
from transformers import pipeline
from transformers import AutoModelForSeq2SeqLM, Seq2SeqTrainingArguments, Seq2SeqTrainer, DataCollatorForSeq2Seq
from datasets import Dataset
import huggingface_hub
from huggingface_hub import HfFolder
from transformers import EarlyStoppingCallback

huggingface_hub.constants.HF_HUB_DOWNLOAD_TIMEOUT = 300  # 5 minutes

# Set up directory paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "..", "models")
TRANSFORMER_MODEL_DIR = os.path.join(MODELS_DIR, "transformer", "models/transformer/transformer_model")
SEQ2SEQ_MODEL_DIR = os.path.join(MODELS_DIR, "seq2seq", "models/seq2seq/seq2seq_model")
DATA_DIR = os.path.join(BASE_DIR, "..", "data")
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, "processed")
RAW_DATA_DIR = os.path.join(DATA_DIR, "raw")

# Create directories if they don't exist
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(TRANSFORMER_MODEL_DIR, exist_ok=True)
os.makedirs(SEQ2SEQ_MODEL_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
os.makedirs(RAW_DATA_DIR, exist_ok=True)

def preprocess_for_transformer(data_file, output_dir="data/processed"):
    """
    Preprocess the scraped data for transformer model training with enhanced data augmentation
    
    Args:
        data_file: Path to scraped data file (CSV or JSON)
        output_dir: Directory to save processed data
    
    Returns:
        Processed dataset ready for transformer model
    """
    print(f"Loading data from {data_file}...")
    
    # Load the data
    if data_file.endswith('.csv'):
        data = pd.read_csv(data_file)
    elif data_file.endswith('.json'):
        data = pd.read_json(data_file)
    else:
        raise ValueError("Data file must be CSV or JSON")
    
    # Create directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Clean and expand data
    processed_rows = []
    
    print(f"Processing {len(data)} commands...")
    
    # Common command patterns and their variations
    command_patterns = {
        'dir': [
            "List files in the current directory",
            "Show me what files are in this folder",
            "Display directory contents",
            "What's in this directory",
            "Show folder contents",
            "View files in this location",
            "Display the contents of this folder",
            "Show me the files here",
            "List the contents of this directory",
            "What files are in this folder"
        ],
        'cd': [
            "Change to a different directory",
            "Move to another folder",
            "Switch directories",
            "Navigate to directory",
            "Go to folder",
            "Change the current directory",
            "Move to a different location",
            "Switch to another folder",
            "Navigate to a different directory",
            "Change working directory"
        ],
        'copy': [
            "Copy a file from one location to another",
            "Make a copy of a file",
            "Duplicate a file",
            "Copy files",
            "Create a copy of a file",
            "Duplicate files to another location",
            "Make a backup of a file",
            "Copy a file to a new location",
            "Create a duplicate of a file",
            "Make a copy of files"
        ],
        'del': [
            "Delete a file",
            "Remove a file",
            "Get rid of a file",
            "Erase files",
            "Remove files from the system",
            "Delete files permanently",
            "Remove files from disk",
            "Erase files from the system",
            "Delete files from the computer",
            "Remove files from storage"
        ],
        'ipconfig': [
            "Show my IP address",
            "What's my network configuration",
            "Display network settings",
            "Check IP configuration",
            "View network information",
            "Show network adapter details",
            "Display IP configuration",
            "Check network settings",
            "View IP address information",
            "Show network configuration"
        ]
    }
    
    for _, row in data.iterrows():
        # Basic cleaning
        command = row['command']
        description = re.sub(r'\s+', ' ', row['description']).strip()
        description = re.sub(r'<.*?>', '', description)
        
        # Create base variations
        variations = [
            f"How do I {description.lower().rstrip('.')}?",
            f"Show me how to {description.lower().rstrip('.')}",
            f"What command {description.lower().rstrip('.')}?",
            f"I need to {description.lower().rstrip('.')}",
            f"Command to {description.lower().rstrip('.')}",
            f"How to {description.lower().rstrip('.')}",
            f"Windows command for {description.lower().rstrip('.')}",
            f"CLI command to {description.lower().rstrip('.')}",
            f"Terminal command for {description.lower().rstrip('.')}",
            f"Command prompt command to {description.lower().rstrip('.')}"
        ]
        
        # Add command-specific variations if available
        if command.lower() in command_patterns:
            variations.extend(command_patterns[command.lower()])
        
        # Add each variation as a training example
        for query in variations:
            processed_rows.append({
                'text': query,
                'label': command
            })
    
    # Convert to DataFrame
    processed_df = pd.DataFrame(processed_rows)
    
    # Save processed data
    processed_df.to_csv(os.path.join(output_dir, "data/processed.csv"), index=False)
    
    # Get unique labels and create label mapping
    labels = processed_df['label'].unique().tolist()
    label_to_id = {label: i for i, label in enumerate(labels)}
    id_to_label = {i: label for i, label in enumerate(labels)}
    
    # Save label mappings
    with open(os.path.join(output_dir, "label_mapping.json"), 'w') as f:
        json.dump({'label_to_id': label_to_id, 'id_to_label': id_to_label}, f)
    
    # Apply label mapping
    processed_df['label_id'] = processed_df['label'].map(label_to_id)
    
    print(f"Processed {len(processed_df)} training examples")
    print(f"Found {len(labels)} unique commands")
    
    return processed_df, label_to_id, id_to_label

def train_classification_transformer(data, label_to_id, id_to_label, model_name="roberta-base", output_dir=TRANSFORMER_MODEL_DIR, epochs=5):
    """
    Train an improved transformer classification model
    
    Args:
        data: DataFrame with processed data
        label_to_id: Mapping from label names to IDs
        id_to_label: Mapping from IDs to label names
        model_name: Pretrained model to use
        output_dir: Directory to save the model
        epochs: Number of training epochs
    
    Returns:
        Trained model and tokenizer
    """
    print(f"Training improved classification transformer using {model_name}...")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Calculate minimum examples needed per class
    min_examples_per_class = 5  # Minimum examples needed for stratification
    
    # Filter classes with enough examples
    class_counts = data['label'].value_counts()
    valid_classes = class_counts[class_counts >= min_examples_per_class].index.tolist()
    
    if len(valid_classes) < len(class_counts):
        print(f"Warning: {len(class_counts) - len(valid_classes)} classes have fewer than {min_examples_per_class} examples")
        print("These classes will be excluded from training")
    
    # Filter data to only include valid classes
    filtered_data = data[data['label'].isin(valid_classes)]
    
    # Split the data with stratification for valid classes
    train_df, eval_df = train_test_split(
        filtered_data, 
        test_size=0.2, 
        random_state=42,
        stratify=filtered_data['label']
    )
    
    # Save label mappings in the model directory
    with open(os.path.join(output_dir, "label_mapping.json"), 'w') as f:
        json.dump({'label_to_id': label_to_id, 'id_to_label': id_to_label}, f)
    
    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, 
        num_labels=len(label_to_id),
        problem_type="single_label_classification"
    )
    
    # Convert to HuggingFace datasets
    train_dataset = Dataset.from_pandas(train_df)
    eval_dataset = Dataset.from_pandas(eval_df)
    
    # Ensure labels are integers
    train_dataset = train_dataset.map(
        lambda example: {"label": label_to_id[example["label"]]},
        batched=False
    )
    eval_dataset = eval_dataset.map(
        lambda example: {"label": label_to_id[example["label"]]},
        batched=False
    )
    
    # Enhanced tokenization function with dynamic padding
    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            padding="max_length",
            truncation=True,
            max_length=256,  # Increased max length
            return_tensors="pt"
        )
    
    # Tokenize datasets
    train_dataset = train_dataset.map(tokenize_function, batched=True)
    eval_dataset = eval_dataset.map(tokenize_function, batched=True)
    
    # Enhanced compute_metrics function
    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=1)
        accuracy = np.mean(predictions == labels)
        
        # Calculate precision, recall, F1 for each class
        from sklearn.metrics import precision_recall_fscore_support
        precision, recall, f1, _ = precision_recall_fscore_support(
            labels, predictions, average='weighted'
        )
        
        return {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1
        }
    
    # Enhanced training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        learning_rate=2e-5,  # Lower learning rate
        per_device_train_batch_size=8,  # Smaller batch size
        per_device_eval_batch_size=8,
        num_train_epochs=epochs,
        weight_decay=0.01,
        load_best_model_at_end=True,
        metric_for_best_model="eval_f1",  # Use F1 score for model selection
        warmup_ratio=0.1,  # Add warmup
        lr_scheduler_type="cosine",  # Use cosine learning rate schedule
        logging_steps=10,
        save_total_limit=3,
        fp16=torch.cuda.is_available(),  # Use mixed precision if available
        gradient_accumulation_steps=4,  # Accumulate gradients
    )
    
    # Set up trainer with early stopping
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]  # Add early stopping
    )
    
    # Train the model
    trainer.train()
    
    # Save the model
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    
    print(f"Model saved to {output_dir}")
    
    return model, tokenizer


def train_seq2seq_transformer(data, model_name="t5-base", output_dir=SEQ2SEQ_MODEL_DIR, epochs=5):
    """
    Train an improved sequence-to-sequence transformer model
    
    Args:
        data: DataFrame with processed data
        model_name: Pretrained model to use
        output_dir: Directory to save the model
        epochs: Number of training epochs
    
    Returns:
        Trained model and tokenizer
    """
    print(f"Training improved sequence-to-sequence transformer using {model_name}...")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Enhanced data preparation for seq2seq
    seq2seq_data = []
    for _, row in data.iterrows():
        # Add multiple variations of the input text
        input_variations = [
            f"translate query to command: {row['text']}",
            f"convert to windows command: {row['text']}",
            f"what command: {row['text']}",
            f"windows command for: {row['text']}"
        ]
        
        for input_text in input_variations:
            seq2seq_data.append({
                'input_text': input_text,
                'target_text': row['label']
            })
    
    seq2seq_df = pd.DataFrame(seq2seq_data)
    
    # Split the data with stratification
    train_df, eval_df = train_test_split(
        seq2seq_df, 
        test_size=0.2, 
        random_state=42,
        stratify=seq2seq_df['target_text']
    )
    
    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    
    # Convert to HuggingFace datasets
    train_dataset = Dataset.from_pandas(train_df)
    eval_dataset = Dataset.from_pandas(eval_df)
    
    # Enhanced tokenization function
    def tokenize_function(examples):
        # Tokenize inputs
        model_inputs = tokenizer(
            examples["input_text"], 
            max_length=256,  # Increased max length
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        
        # Tokenize targets
        with tokenizer.as_target_tokenizer():
            labels = tokenizer(
                examples["target_text"], 
                max_length=64,
                padding="max_length",
                truncation=True,
                return_tensors="pt"
            )
            
        # Replace padding token id's of the labels by -100
        labels["input_ids"] = [
            [(l if l != tokenizer.pad_token_id else -100) for l in label] 
            for label in labels["input_ids"]
        ]
        
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs
    
    # Tokenize datasets
    train_dataset = train_dataset.map(tokenize_function, batched=True)
    eval_dataset = eval_dataset.map(tokenize_function, batched=True)
    
    # Data collator
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        label_pad_token_id=-100
    )
    
    # Enhanced training arguments
    training_args = Seq2SeqTrainingArguments(
        output_dir=output_dir,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        learning_rate=3e-5,  # Adjusted learning rate
        per_device_train_batch_size=8,  # Smaller batch size
        per_device_eval_batch_size=8,
        weight_decay=0.01,
        save_total_limit=3,
        num_train_epochs=epochs,
        predict_with_generate=True,
        fp16=torch.cuda.is_available(),  # Use mixed precision if available
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",  # Use loss for model selection
        warmup_ratio=0.1,  # Add warmup
        lr_scheduler_type="cosine",  # Use cosine learning rate schedule
        logging_steps=10,
        gradient_accumulation_steps=4,  # Accumulate gradients
        generation_max_length=64,
        generation_num_beams=4,
    )
    
    # Set up trainer with early stopping
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]  # Add early stopping
    )
    
    # Train the model
    trainer.train()
    
    # Save the model
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    
    print(f"Model saved to {output_dir}")
    
    return model, tokenizer

def predict_with_classification_model(query, model_dir=TRANSFORMER_MODEL_DIR):
    """
    Predict a command using the classification transformer model
    
    Args:
        query: Natural language query
        model_dir: Directory where the model is saved
    
    Returns:
        Predicted command and confidence
    """
    # Load label mapping
    with open(os.path.join(model_dir, "label_mapping.json"), 'r') as f:
        mapping = json.load(f)
        id_to_label = mapping['id_to_label']
        # Convert string keys back to integers
        id_to_label = {int(k): v for k, v in id_to_label.items()}
    
    # Create pipeline
    classifier = pipeline(
        "text-classification",
        model=model_dir,
        tokenizer=model_dir,
        return_all_scores=True
    )
    
    # Get prediction
    result = classifier(query)
    scores = result[0]
    
    # Sort by score
    scores.sort(key=lambda x: x['score'], reverse=True)
    
    # Get top prediction
    top_pred = scores[0]
    label_id = int(top_pred['label'].split('_')[-1])
    command = id_to_label[label_id]
    confidence = top_pred['score']
    
    return command, confidence

def predict_with_models/seq2seq/seq2seq_model(query, model_dir=SEQ2SEQ_MODEL_DIR):
    """
    Predict a command using the improved sequence-to-sequence transformer model
    """
    # Load model and tokenizer
    model = AutoModelForSeq2SeqLM.from_pretrained(model_dir)
    tokenizer = AutoTokenizer.from_pretrained(model_dir)

    # Prepare multiple input variations
    input_variations = [
        f"translate query to command: {query}",
        f"convert to windows command: {query}",
        f"what command: {query}",
        f"windows command for: {query}"
    ]
    
    # Generate predictions for each variation
    predictions = []
    for input_text in input_variations:
        inputs = tokenizer(input_text, return_tensors="pt", max_length=256, 
                          padding="max_length", truncation=True)
        
        # Generate with beam search
        outputs = model.generate(
            **inputs,
            max_length=64,
            min_length=1,
            num_beams=5,  # Increased beam size
            num_return_sequences=3,  # Get multiple candidates
            do_sample=True,
            temperature=0.7,
            top_k=50,
            top_p=0.95,
            no_repeat_ngram_size=2,
            early_stopping=True
        )
        
        # Decode predictions
        for output in outputs:
            prediction = tokenizer.decode(output, skip_special_tokens=True)
            if prediction and prediction.strip():
                predictions.append(prediction)
    
    # Get unique predictions
    unique_predictions = list(set(predictions))
    
    # If no valid predictions, use fallback
    if not unique_predictions:
        # Try the most common commands as fallback
        common_commands = {
            "directory": "dir",
            "list files": "dir",
            "file list": "dir",
            "change directory": "cd",
            "move folder": "cd",
            "ip": "ipconfig",
            "network": "ipconfig",
            "delete": "del",
            "remove": "del",
            "copy": "copy"
        }
        
        # Check for keywords in the query
        for keyword, cmd in common_commands.items():
            if keyword.lower() in query.lower():
                return cmd, 0.5
        
        return "No command prediction generated", 0.0
    
    # Return the most frequent prediction
    from collections import Counter
    most_common = Counter(unique_predictions).most_common(1)[0]
    return most_common[0], 1.0

def create_transformer_cli(command_data_file=None, model_type="classification", model_dir=None):
    """
    Create an interactive CLI using the transformer model
    
    Args:
        command_data_file: File with detailed command information (optional)
        model_type: Type of model to use ('classification' or 'seq2seq')
        model_dir: Directory where the model is saved
    """
    print("\nNatural Language to Windows Command System (Transformer Edition)")
    print("Type 'exit' to quit\n")
    
    try:
        # Set default model directory if not provided
        if model_dir is None:
            model_dir = TRANSFORMER_MODEL_DIR if model_type == "classification" else SEQ2SEQ_MODEL_DIR
        
        # Check if model exists
        if not os.path.exists(model_dir):
            print(f"Error: Model directory '{model_dir}' not found!")
            return
        
        # Load command details if available
        command_dict = {}
        if command_data_file:
            try:
                if command_data_file.endswith('.csv'):
                    df = pd.read_csv(command_data_file)
                elif command_data_file.endswith('.json'):
                    df = pd.read_json(command_data_file)
                else:
                    print("Warning: Unsupported file format. Command details won't be available.")
                    df = None
                    
                if df is not None:
                    command_dict = {row['command']: row for _, row in df.iterrows()}
            except Exception as e:
                print(f"Warning: Could not load command details: {str(e)}")
        
        # Choose prediction function based on model type
        predict_func = predict_with_classification_model if model_type == "classification" else predict_with_models/seq2seq/seq2seq_model
        
        while True:
            try:
                user_input = input("What would you like to do? ").strip()
                
                if not user_input:
                    continue
                    
                if user_input.lower() in ['exit', 'quit', 'q']:
                    print("Goodbye!")
                    break
                
                # Predict the command
                command, confidence = predict_func(user_input, model_dir)
                
                # Display information
                print(f"\nSuggested command: {command}")
                if model_type == "classification":
                    print(f"Confidence: {confidence:.2f}")
                
                if command in command_dict:
                    cmd_info = command_dict[command]
                    print(f"\nDescription: {cmd_info['description']}")
                    
                    if 'syntax' in cmd_info and cmd_info['syntax']:
                        print(f"\nSyntax:")
                        print(cmd_info['syntax'])
                    
                    if 'examples' in cmd_info and isinstance(cmd_info['examples'], list) and len(cmd_info['examples']) > 0:
                        print("\nExample:")
                        print(f"  {cmd_info['examples'][0]}")
                
            except KeyboardInterrupt:
                print("\nOperation cancelled. Type 'exit' to quit.")
            except Exception as e:
                print(f"Error: {str(e)}")
            
            print("\n" + "-"*50)
            
    except Exception as e:
        print(f"Fatal error: {str(e)}")


def main():
    """
    Main function to run the transformer model training pipeline
    """
    print("Advanced Windows Command NLP Transformer Model")
    print("=============================================")
    
    try:
        # Get input file from user
        data_file = input("Enter path to your scraped data file (CSV or JSON): ").strip()
        
        # Check if the file exists
        if not os.path.exists(data_file):
            print(f"Error: File '{data_file}' not found!")
            return
        
        # Validate file extension
        if not (data_file.endswith('.csv') or data_file.endswith('.json')):
            print("Error: File must be in CSV or JSON format!")
            return
        
        # Preprocess data
        print("\nPreprocessing data...")
        data, label_to_id, id_to_label = preprocess_for_transformer(data_file)
        
        # Choose model type
        print("\nSelect model type:")
        print("1. Classification Transformer (DistilBERT) - Faster, good accuracy")
        print("2. Sequence-to-Sequence Transformer (T5) - Better for complex queries")
        
        while True:
            model_choice = input("Enter choice (1 or 2): ").strip()
            if model_choice in ["1", "2"]:
                break
            print("Invalid choice. Please enter 1 or 2.")
        
        # Train selected model
        print("\nTraining model...")
        if model_choice == "1":
            # Train classification model
            model, tokenizer = train_classification_transformer(data, label_to_id, id_to_label)
            model_type = "classification"
            model_dir = TRANSFORMER_MODEL_DIR
        else:
            # Train sequence-to-sequence model
            model, tokenizer = train_seq2seq_transformer(data)
            model_type = "seq2seq"
            model_dir = SEQ2SEQ_MODEL_DIR
        
        print("\nModel training completed successfully!")
        print("Starting interactive CLI...\n")
        
        # Start the CLI for testing
        create_transformer_cli(data_file, model_type, model_dir)
        
    except KeyboardInterrupt:
        print("\n\nOperation cancelled by user.")
    except Exception as e:
        print(f"\nAn error occurred: {str(e)}")
        print("Please check your input and try again.")

if __name__ == "__main__":
    main()