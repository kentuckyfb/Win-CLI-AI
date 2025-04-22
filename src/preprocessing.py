import json
from pathlib import Path
import re

def clean_text(text):
    """Clean and normalize text data"""
    if not text:
        return ""
    # Remove special characters and extra whitespace
    text = re.sub(r'[^\w\s-]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text.lower()

# Load all data files
data_folder = Path("data")
with open(data_folder / "data/processed.json") as f:
    data/processed = json.load(f)
    
with open(data_folder / "windows_commands_basic.json") as f:
    basic_commands = json.load(f)
    
with open(data_folder / "windows_commands_detailed.json") as f:
    detailed_commands = json.load(f)

# Create a mapping of command to all its information
command_info = {cmd["command"]: cmd for cmd in basic_commands}

# Add detailed info to the command mapping
for cmd in detailed_commands:
    command_name = cmd["command"]
    if command_name in command_info:
        command_info[command_name].update(cmd)

# Process and combine all data
data/processed = []
for item in data/processed:
    command_name = item["command"]
    if command_name not in command_info:
        continue
        
    # Get all available info for this command
    cmd_data = command_info[command_name]
    
    # Create enhanced training sample
    processed_item = {
        "query": clean_text(item["query"]),
        "command": command_name,
        "description": clean_text(cmd_data.get("description", "")),
        "description_detailed": clean_text(cmd_data.get("description_detailed", "")),
        "syntax": clean_text(cmd_data.get("syntax", "")),
        "examples": [clean_text(ex) for ex in cmd_data.get("examples", [])],
        "url": cmd_data.get("url", "")
    }
    
    # Create a combined text field for semantic search
    combined_text = " ".join([
        processed_item["query"],
        processed_item["description"],
        processed_item["description_detailed"],
        processed_item["syntax"],
        " ".join(processed_item["examples"])
    ])
    processed_item["combined_text"] = clean_text(combined_text)
    
    data/processed.append(processed_item)

# Save processed data
output_path = data_folder / "data.json"
with open(output_path, "w") as f:
    json.dump(data/processed, f, indent=2)

print(f"Processed data saved to {output_path}")
print(f"Total samples: {len(data/processed)}")
print(f"Unique commands: {len(command_info)}")