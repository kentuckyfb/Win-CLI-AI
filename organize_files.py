import os
import shutil

def create_directory_structure():
    """Create the new directory structure"""
    directories = [
        "models",
        "models/transformer",
        "models/classifier",
        "models/seq2seq",
        "models/hybrid",
        "data",
        "data/raw",
        "data/processed",
        "src"
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"Created directory: {directory}")

def move_files():
    """Move files to their new locations"""
    # Move Python files to src
    python_files = [f for f in os.listdir('.') if f.endswith('.py')]
    for file in python_files:
        if file != 'organize_files.py':  # Don't move this script
            shutil.move(file, os.path.join('src', file))
            print(f"Moved {file} to src/")
    
    # Move model directories to appropriate subdirectories
    model_mappings = {
        'transformer_model': 'models/transformer',
        'seq2seq_model': 'models/seq2seq',
        'command_classifier': 'models/classifier',
        'hybrid_model': 'models/hybrid'
    }
    
    for old_dir, new_dir in model_mappings.items():
        if os.path.exists(old_dir):
            shutil.move(old_dir, new_dir)
            print(f"Moved {old_dir} to {new_dir}/")
    
    # Move any other model directories to models
    for item in os.listdir('.'):
        if os.path.isdir(item) and item not in model_mappings and item not in ['models', 'data', 'src']:
            shutil.move(item, os.path.join('models', item))
            print(f"Moved {item} to models/")
    
    # Move data files to data/raw
    data_files = [f for f in os.listdir('.') if f.endswith(('.csv', '.json', '.txt'))]
    for file in data_files:
        shutil.move(file, os.path.join('data/raw', file))
        print(f"Moved {file} to data/raw/")
    
    # Move processed data to data/processed
    processed_dirs = ['transformer_data', 'processed_data', 'training_data']
    for dir_name in processed_dirs:
        if os.path.exists(dir_name):
            for file in os.listdir(dir_name):
                shutil.move(
                    os.path.join(dir_name, file),
                    os.path.join('data/processed', file)
                )
            os.rmdir(dir_name)
            print(f"Moved contents of {dir_name} to data/processed/")

def update_file_paths():
    """Update file paths in Python files"""
    src_dir = os.path.join(os.getcwd(), 'src')
    for file in os.listdir(src_dir):
        if file.endswith('.py'):
            file_path = os.path.join(src_dir, file)
            with open(file_path, 'r') as f:
                content = f.read()
            
            # Update paths
            content = content.replace('transformer_model', 'models/transformer/transformer_model')
            content = content.replace('seq2seq_model', 'models/seq2seq/seq2seq_model')
            content = content.replace('command_classifier', 'models/classifier/command_classifier')
            content = content.replace('hybrid_model', 'models/hybrid/hybrid_model')
            content = content.replace('transformer_data', 'data/processed')
            content = content.replace('processed_data', 'data/processed')
            content = content.replace('training_data', 'data/processed')
            
            with open(file_path, 'w') as f:
                f.write(content)
            
            print(f"Updated paths in {file}")

def main():
    """Main function to organize files"""
    print("Organizing files into new directory structure...")
    
    # Create directory structure
    create_directory_structure()
    
    # Move files
    move_files()
    
    # Update file paths
    update_file_paths()
    
    print("\nDirectory structure organized successfully!")
    print("New structure:")
    print("├── models/")
    print("│   ├── transformer/")
    print("│   │   └── transformer_model/")
    print("│   ├── classifier/")
    print("│   │   └── command_classifier/")
    print("│   ├── seq2seq/")
    print("│   │   └── seq2seq_model/")
    print("│   └── hybrid/")
    print("│       └── hybrid_model/")
    print("├── data/")
    print("│   ├── raw/")
    print("│   └── processed/")
    print("└── src/")
    print("    ├── transformer_cmd_model.py")
    print("    └── run_models.py")

if __name__ == "__main__":
    main() 