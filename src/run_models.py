import os
import json
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModelForSeq2SeqLM, pipeline
import torch

# Set up directory paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "..", "models")
TRANSFORMER_MODEL_DIR = os.path.join(MODELS_DIR, "transformer", "models/transformer/transformer_model")
SEQ2SEQ_MODEL_DIR = os.path.join(MODELS_DIR, "seq2seq", "models/seq2seq/seq2seq_model")
CLASSIFIER_MODEL_DIR = os.path.join(MODELS_DIR, "classifier", "models/classifier/command_classifier")
HYBRID_MODEL_DIR = os.path.join(MODELS_DIR, "hybrid", "models/hybrid/hybrid_model")
DATA_DIR = os.path.join(BASE_DIR, "..", "data")
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, "processed")
RAW_DATA_DIR = os.path.join(DATA_DIR, "raw")

def load_models():
    """Load all available models"""
    print("Loading models...")
    
    models = {}
    
    # Load classification model if available
    if os.path.exists(TRANSFORMER_MODEL_DIR):
        models['transformer'] = pipeline(
            "text-classification",
            model=TRANSFORMER_MODEL_DIR,
            tokenizer=TRANSFORMER_MODEL_DIR,
            return_all_scores=True
        )
    
    # Load seq2seq model if available
    if os.path.exists(SEQ2SEQ_MODEL_DIR):
        models['seq2seq'] = {
            'model': AutoModelForSeq2SeqLM.from_pretrained(SEQ2SEQ_MODEL_DIR),
            'tokenizer': AutoTokenizer.from_pretrained(SEQ2SEQ_MODEL_DIR)
        }
    
    # Load command classifier if available
    if os.path.exists(CLASSIFIER_MODEL_DIR):
        models['classifier'] = pipeline(
            "text-classification",
            model=CLASSIFIER_MODEL_DIR,
            tokenizer=CLASSIFIER_MODEL_DIR,
            return_all_scores=True
        )
    
    # Load hybrid model if available
    if os.path.exists(HYBRID_MODEL_DIR):
        models['hybrid'] = {
            'model': AutoModelForSeq2SeqLM.from_pretrained(HYBRID_MODEL_DIR),
            'tokenizer': AutoTokenizer.from_pretrained(HYBRID_MODEL_DIR)
        }
    
    # Load label mappings
    label_mappings = {}
    for model_dir in [TRANSFORMER_MODEL_DIR, CLASSIFIER_MODEL_DIR]:
        mapping_file = os.path.join(model_dir, "label_mapping.json")
        if os.path.exists(mapping_file):
            with open(mapping_file, 'r') as f:
                mapping = json.load(f)
                label_mappings[os.path.basename(model_dir)] = {
                    int(k): v for k, v in mapping['id_to_label'].items()
                }
    
    return models, label_mappings

def predict_with_classification(query, classifier, id_to_label):
    """Predict using classification model"""
    result = classifier(query)
    scores = result[0]
    scores.sort(key=lambda x: x['score'], reverse=True)
    
    # Get top 3 predictions
    top_predictions = []
    for i in range(min(3, len(scores))):
        label_id = int(scores[i]['label'].split('_')[-1])
        command = id_to_label[label_id]
        confidence = scores[i]['score']
        top_predictions.append((command, confidence))
    
    return top_predictions

def predict_with_seq2seq(query, model, tokenizer):
    """Predict using seq2seq model"""
    input_text = f"translate query to command: {query}"
    inputs = tokenizer(input_text, return_tensors="pt", max_length=128, 
                      padding="max_length", truncation=True)
    
    outputs = model.generate(
        **inputs,
        max_length=64,
        min_length=1,
        num_beams=4,
        do_sample=True,
        temperature=0.7,
        top_k=50,
        top_p=0.95,
        no_repeat_ngram_size=2,
        early_stopping=True
    )
    
    prediction = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return prediction

def main():
    """Main function to run the CLI"""
    print("\nWindows Command Prediction System")
    print("=================================")
    print("This system uses two AI models to predict Windows commands:")
    print("1. Classification Model - Predicts the most likely command")
    print("2. Seq2Seq Model - Generates command based on your query")
    print("\nType 'exit' to quit\n")
    
    try:
        # Load models
        models, label_mappings = load_models()
        print("Models loaded successfully!\n")
        
        while True:
            query = input("What would you like to do? ").strip()
            
            if not query:
                continue
                
            if query.lower() in ['exit', 'quit', 'q']:
                print("Goodbye!")
                break
            
            try:
                # Get predictions from both models
                print("\nClassification Model Predictions:")
                class_predictions = predict_with_classification(query, models['classifier'], label_mappings['classifier'])
                for i, (command, confidence) in enumerate(class_predictions, 1):
                    print(f"{i}. {command} (confidence: {confidence:.2f})")
                
                print("\nSeq2Seq Model Prediction:")
                seq2seq_prediction = predict_with_seq2seq(query, models['seq2seq']['model'], models['seq2seq']['tokenizer'])
                print(f"Generated command: {seq2seq_prediction}")
                
            except Exception as e:
                print(f"Error making prediction: {str(e)}")
            
            print("\n" + "-"*50)
            
    except Exception as e:
        print(f"Error: {str(e)}")
        print("Please make sure all models are trained before running this script.")

if __name__ == "__main__":
    main() 