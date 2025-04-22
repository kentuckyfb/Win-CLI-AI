# Win-CLI-AI

A collection of AI models designed to understand and generate Windows command-line instructions. This project uses various transformer-based models to classify and generate Windows commands based on natural language input.

## Project Structure

```
Windows CLI AI/
├── models/                  # All AI models
│   ├── transformer/        # Transformer model for command classification
│   │   └── transformer_model/
│   ├── classifier/         # Command classifier model
│   │   └── command_classifier/
│   ├── seq2seq/           # Sequence-to-sequence model for command generation
│   │   └── seq2seq_model/
│   └── hybrid/            # Hybrid model combining multiple approaches
│       └── hybrid_model/
├── data/                   # Data files
│   ├── raw/               # Raw scraped data
│   └── processed/         # Processed training data
├── src/                    # Source code
│   ├── transformer_cmd_model.py  # Main model training and prediction code
│   ├── run_models.py      # CLI interface for model interaction
│   └── organize_files.py  # Script for organizing project structure
└── requirements.txt        # Project dependencies
```

## Data Collection and Processing

### Data Sources
- Windows Command Line Documentation
- Stack Overflow Q&A related to Windows commands
- Microsoft TechNet articles
- Windows PowerShell documentation
- Common Windows command usage examples

### Data Extraction
The raw data is collected from various sources and processed into structured formats:
- Command descriptions and examples
- Natural language queries and their corresponding commands
- Command syntax and parameters
- Common use cases and scenarios

### Data Processing
1. Raw data is stored in `data/raw/` directory
2. Data is preprocessed and cleaned
3. Processed data is stored in `data/processed/` directory
4. Data is split into training, validation, and test sets

## Models

### 1. Transformer Model (Classification)
- Location: `models/transformer/transformer_model/`
- Purpose: Classifies natural language input into Windows command categories
- Based on: RoBERTa architecture
- Features:
  - Handles multiple command categories
  - Provides confidence scores
  - Supports command parameter prediction

### 2. Command Classifier
- Location: `models/classifier/command_classifier/`
- Purpose: Specialized classification for specific command types
- Features:
  - Fine-grained command classification
  - Parameter extraction
  - Syntax validation

### 3. Seq2Seq Model
- Location: `models/seq2seq/seq2seq_model/`
- Purpose: Generates Windows commands from natural language descriptions
- Based on: T5 architecture
- Features:
  - End-to-end command generation
  - Parameter suggestion
  - Multiple command suggestions

### 4. Hybrid Model
- Location: `models/hybrid/hybrid_model/`
- Purpose: Combines classification and generation capabilities
- Features:
  - Multi-stage processing
  - Enhanced accuracy
  - Better handling of complex queries

## Setup and Installation

1. Clone the repository:
```bash
git clone https://github.com/kentuckyfb/Win-CLI-AI.git
cd Win-CLI-AI
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Organize the project structure:
```bash
python src/organize_files.py
```

## Usage

1. Train the models:
```bash
python src/transformer_cmd_model.py --train
```

2. Run the CLI interface:
```bash
python src/run_models.py
```

3. Interact with the models:
```
Enter your query: How do I list all files in a directory?
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Hugging Face Transformers library
- Microsoft Windows documentation
- Open-source community contributions 