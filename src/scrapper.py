import requests
from bs4 import BeautifulSoup
import pandas as pd
import time
import re
import os
from urllib.parse import urljoin

def scrape_command_list(url='https://ss64.com/nt/'):
    """Scrape the main command list page to get commands and brief descriptions"""
    print(f"Scraping command list from {url}")
    response = requests.get(url)
    soup = BeautifulSoup(response.content, 'html.parser')
    
    commands_data = []
    
    # The commands are in table rows where the second cell contains a link
    rows = soup.find_all('tr')
    
    for row in rows:
        # Skip header rows or rows that don't have enough cells
        cells = row.find_all('td')
        if len(cells) < 3:
            continue
        
        # The second cell contains the command name with link
        command_cell = cells[1]
        command_link = command_cell.find('a')
        
        # Skip rows without a command link
        if not command_link:
            continue
        
        command_name = command_link.text.strip()
        command_url = urljoin(url, command_link['href'])
        
        # The third cell contains the brief description
        description = cells[2].text.strip()
        
        commands_data.append({
            'command': command_name,
            'description': description,
            'url': command_url
        })
    
    print(f"Found {len(commands_data)} commands")
    return commands_data

def scrape_command_details(command_data, delay=1):
    """
    Scrape detailed information for each command
    
    Args:
        command_data: List of dictionaries with command info
        delay: Delay between requests to avoid overloading the server
    
    Returns:
        List of enriched command dictionaries with detailed info
    """
    detailed_data = []
    
    for i, cmd in enumerate(command_data):
        print(f"Scraping details for {cmd['command']} ({i+1}/{len(command_data)})")
        
        try:
            # Add a delay to be respectful to the server
            time.sleep(delay)
            
            response = requests.get(cmd['url'])
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Get the syntax section - usually in a <pre> tag
            syntax_section = soup.find('pre')
            syntax = syntax_section.text.strip() if syntax_section else "No syntax information available"
            
            # Get example section - often follows an h3 with 'Examples'
            examples = []
            example_header = soup.find(lambda tag: tag.name == 'h3' and 'Example' in tag.text)
            if example_header:
                current = example_header.next_sibling
                while current and not (current.name == 'h3'):
                    if current.name == 'blockquote':
                        code = current.find(class_='code')
                        if code:
                            examples.append(code.text.strip())
                    current = current.next_sibling
            
            # Get detailed description - typically paragraphs after h1 and before pre
            description_detailed = ""
            h1 = soup.find('h1')
            if h1:
                current = h1.next_sibling
                while current and current.name != 'pre':
                    if current.name == 'p':
                        description_detailed += current.text.strip() + " "
                    current = current.next_sibling
            
            # Add all the detailed information to our command data
            cmd_detailed = cmd.copy()
            cmd_detailed.update({
                'syntax': syntax,
                'examples': examples,
                'description_detailed': description_detailed.strip()
            })
            
            detailed_data.append(cmd_detailed)
            
        except Exception as e:
            print(f"Error scraping {cmd['command']}: {str(e)}")
            # Still add the command to our list, just without the extra details
            detailed_data.append(cmd)
    
    return detailed_data

def save_to_csv(data, filename='windows_commands.csv'):
    """Save the scraped data to a CSV file"""
    df = pd.DataFrame(data)
    df.to_csv(filename, index=False, encoding='utf-8')
    print(f"Data saved to {filename}")
    return filename

def save_to_json(data, filename='windows_commands.json'):
    """Save the scraped data to a JSON file"""
    df = pd.DataFrame(data)
    df.to_json(filename, orient='records', indent=2)
    print(f"Data saved to {filename}")
    return filename

def create_data/processed(data):
    """
    Format the data into training examples for an AI model.
    Each example pairs a natural language query with the appropriate command.
    """
    training_pairs = []
    
    for cmd in data:
        # Basic query based on the description
        query = f"How do I {cmd['description'].lower().rstrip('.')}?"
        training_pairs.append({
            'query': query,
            'command': cmd['command']
        })
        
        # More variations
        if 'list' in cmd['description'].lower():
            training_pairs.append({
                'query': f"Show me {cmd['description'].lower().rstrip('.')}",
                'command': cmd['command']
            })
        
        if 'display' in cmd['description'].lower():
            training_pairs.append({
                'query': f"I want to see {cmd['description'].lower().rstrip('.')}",
                'command': cmd['command']
            })
            
        # Add specific examples for common commands
        if cmd['command'].lower() == 'dir':
            training_pairs.append({
                'query': "List all files in the current directory",
                'command': 'dir'
            })
        
        # And so on for other common commands...
    
    return training_pairs

def main():
    # Create a directory for our data
    os.makedirs('data', exist_ok=True)
    
    # Step 1: Scrape the main command list
    commands = scrape_command_list()
    
    # Save the basic command list
    save_to_csv(commands, 'data/windows_commands_basic.csv')
    save_to_json(commands, 'data/windows_commands_basic.json')
    
    # Step 2: Scrape detailed information for each command
    # For testing, you might want to limit to a few commands first
    # test_commands = commands[:5]  # Just the first 5 commands
    detailed_commands = scrape_command_details(commands)
    
    # Save the detailed command information
    save_to_csv(detailed_commands, 'data/windows_commands_detailed.csv')
    save_to_json(detailed_commands, 'data/windows_commands_detailed.json')
    
    # Step 3: Create training data for AI model
    data/processed = create_data/processed(detailed_commands)
    save_to_csv(data/processed, 'data/data/processed.csv')
    save_to_json(data/processed, 'data/data/processed.json')
    
    print("Scraping and data preparation complete!")

if __name__ == "__main__":
    main()