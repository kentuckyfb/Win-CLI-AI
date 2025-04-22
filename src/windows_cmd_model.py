import pandas as pd
import re
import os
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

def preprocess_for_training(data_file, output_file="data/processed_processed.csv"):
    """
    Preprocess the scraped data for model training
    - Clean text
    - Expand query variations
    """
    print(f"Loading data from {data_file}...")
    
    # Load the data
    if data_file.endswith('.csv'):
        data = pd.read_csv(data_file).to_dict('records')
    elif data_file.endswith('.json'):
        data = pd.read_json(data_file).to_dict('records')
    else:
        raise ValueError("Data file must be CSV or JSON")
    
    data/processed = []
    
    print(f"Processing {len(data)} commands...")
    
    for cmd in data:
        # Basic cleaning
        description = re.sub(r'\s+', ' ', cmd['description']).strip()
        
        # Remove any HTML that might have been captured
        description = re.sub(r'<.*?>', '', description)
        
        # Create multiple variations of natural language queries
        variations = [
            f"How do I {description.lower().rstrip('.')}?",
            f"Show me how to {description.lower().rstrip('.')}",
            f"What command {description.lower().rstrip('.')}?",
            f"I need to {description.lower().rstrip('.')}"
        ]
        
        # Add command-specific variations
        if cmd['command'].lower() in ['dir', 'ls']:
            variations.extend([
                "List files in the current directory",
                "Show me what files are in this folder",
                "Display directory contents",
                "What's in this directory",
                "Show folder contents"
            ])
        
        if cmd['command'].lower() in ['cd', 'chdir']:
            variations.extend([
                "Change to a different directory",
                "Move to another folder",
                "Switch directories",
                "Navigate to directory",
                "Go to folder"
            ])
            
        if cmd['command'].lower() == 'copy':
            variations.extend([
                "Copy a file from one location to another",
                "Make a copy of a file",
                "Duplicate a file",
                "Copy files"
            ])
            
        if cmd['command'].lower() in ['del', 'erase']:
            variations.extend([
                "Delete a file",
                "Remove a file",
                "Get rid of a file",
                "Erase files"
            ])
            
        if cmd['command'].lower() == 'ipconfig':
            variations.extend([
                "Show my IP address",
                "What's my network configuration",
                "Display network settings",
                "Check IP configuration"
            ])
        
        # Add each variation as a training example
        for query in variations:
            data/processed.append({
                'query': query,
                'command': cmd['command'],
                'description': description,
                'syntax': cmd.get('syntax', '')
            })
    
    # Save to CSV
    df = pd.DataFrame(data/processed)
    df.to_csv(output_file, index=False, encoding='utf-8')
    print(f"Processed {len(data/processed)} training examples saved to {output_file}")
    
    return df

def train_command_model(data/processed, model_dir="model"):
    """
    Train a model to predict commands from natural language queries
    
    Args:
        data/processed: DataFrame or path to CSV file with training data
        model_dir: Directory to save the model files
    """
    print("Training model...")
    
    # Load the data if it's a file path
    if isinstance(data/processed, str):
        if data/processed.endswith('.csv'):
            df = pd.read_csv(data/processed)
        else:
            raise ValueError("Training data file must be CSV")
    else:
        df = data/processed
    
    print(f"Using {len(df)} training examples")
    
    # Split features and target
    X = df['query']
    y = df['command']
    
    # Split into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print(f"Training set size: {len(X_train)}, Test set size: {len(X_test)}")
    
    # Convert text to features
    vectorizer = TfidfVectorizer(ngram_range=(1, 3))
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)
    
    # Train a classifier
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train_vec, y_train)
    
    # Evaluate the model
    y_pred = clf.predict(X_test_vec)
    print("\nModel performance:")
    print(classification_report(y_test, y_pred))
    
    # Create model directory if it doesn't exist
    os.makedirs(model_dir, exist_ok=True)
    
    # Save the model and vectorizer
    model_file = os.path.join(model_dir, "command_model.pkl")
    vectorizer_file = os.path.join(model_dir, "command_vectorizer.pkl")
    
    joblib.dump(clf, model_file)
    joblib.dump(vectorizer, vectorizer_file)
    
    print(f"Model saved to {model_file}")
    print(f"Vectorizer saved to {vectorizer_file}")
    
    return clf, vectorizer

def predict_command(query, model_dir="model"):
    """
    Predict a Windows command from a natural language query
    
    Args:
        query: Natural language query
        model_dir: Directory where model files are stored
    
    Returns:
        Predicted command and confidence score
    """
    # Load the model and vectorizer
    model_file = os.path.join(model_dir, "command_model.pkl")
    vectorizer_file = os.path.join(model_dir, "command_vectorizer.pkl")
    
    clf = joblib.load(model_file)
    vectorizer = joblib.load(vectorizer_file)
    
    # Process the query
    query_vec = vectorizer.transform([query])
    
    # Make prediction
    command = clf.predict(query_vec)[0]
    
    # Get confidence
    proba = clf.predict_proba(query_vec)[0]
    confidence = max(proba)
    
    return command, confidence

def create_cli(command_data_file=None, model_dir="model"):
    """
    Create an interactive CLI to test the model
    
    Args:
        command_data_file: File with detailed command information (optional)
        model_dir: Directory where model files are stored
    """
    print("\nNatural Language to Windows Command System")
    print("Type 'exit' to quit\n")
    
    # Load detailed command info if available
    command_dict = {}
    if command_data_file:
        if command_data_file.endswith('.csv'):
            df = pd.read_csv(command_data_file)
        elif command_data_file.endswith('.json'):
            df = pd.read_json(command_data_file)
        else:
            print("Warning: Unsupported file format. Command details won't be available.")
            df = None
            
        if df is not None:
            command_dict = {row['command']: row for _, row in df.iterrows()}
    
    while True:
        user_input = input("What would you like to do? ")
        
        if user_input.lower() in ['exit', 'quit', 'q']:
            print("Goodbye!")
            break
        
        # Predict the command
        try:
            command, confidence = predict_command(user_input, model_dir)
            
            # Display information
            print(f"\nSuggested command: {command}")
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
            
        except Exception as e:
            print(f"Error: {str(e)}")
        
        print("\n" + "-"*50)

def main():
    """
    Main function to run the preprocessing and model training pipeline
    """
    # Get input file from user
    print("Windows Command NLP Model Training")
    print("=================================")
    
    data_file = input("Enter path to your scraped data file (CSV or JSON): ")
    
    # Check if the file exists
    if not os.path.exists(data_file):
        print(f"Error: File '{data_file}' not found!")
        return
    
    # Preprocess data
    data/processed = preprocess_for_training(data_file)
    
    # Train the model
    clf, vectorizer = train_command_model(data/processed)
    
    # Start the CLI for testing
    create_cli(data_file)

if __name__ == "__main__":
    main()