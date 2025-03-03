import os
import json
import re
import argparse  # Add this import at the top

# User-defined project path
project_path = "./Things/KG_Multi_Agent/MKG_Multi_Agent/test1"  # Set this to your desired project path


# Function to extract clean conversation from a conversation history string
def extract_clean_conversation(conversation_history, number_of_last_exchanges=50):
    """
    Extracts the last `number_of_last_exchanges` 'content' entries from a conversation history string
    and formats them into a clean conversation history.
        conversation_history (str): The raw string containing the conversation history.
        number_of_last_exchanges (int): The number of most recent exchanges to extract. Default is 20.
    Returns:
        str: A formatted string with the extracted 'content' entries in order.
    """
    # Find all 'content' entries using regex
    content_matches = re.findall(r'"content":\s*"([^"]+)"', conversation_history)

    # Extract the last `number_of_last_exchanges` entries
    recent_content = content_matches[-number_of_last_exchanges:]

    # Join the extracted content into a clean string, separated by newlines
    return "\n".join(recent_content)

def main(project_path):
    # Define input and output directories based on project path
    input_dir = os.path.join(project_path, "inputs", "conversations")
    output_dir = os.path.join(project_path, "outputs", "conversations")

    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Step 1: Gather all JSON file names in the input directory
    json_files = [file for file in os.listdir(input_dir) if file.endswith('.json')]

    # Step 2: Initialize an empty dictionary to store conversations
    conversations_list = {}

    # Step 3: Process each JSON file
    for json_file in json_files:
        input_file_path = os.path.join(input_dir, json_file)
        
        with open(input_file_path, 'r', encoding='utf-8') as f:
            # Read the entire content of the JSON file as a string
            file_content = f.read()
            
            # Use the extract_clean_conversation function to process the content
            clean_conversation = extract_clean_conversation(file_content)
            
            # Add the result to the conversations_list dictionary with the file name as the key
            conversations_list[json_file] = clean_conversation

    # Step 4: Save the resulting dictionary to a JSON file in the output directory
    output_file_path = os.path.join(output_dir, "conversations_list.json")
    with open(output_file_path, 'w', encoding='utf-8') as f:
        json.dump(conversations_list, f, indent=4, ensure_ascii=False)

    print(f"Conversations list saved successfully at: {output_file_path}")

if __name__ == "__main__":
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Process conversation JSON files.')
    parser.add_argument('--project_path', type=str, default="./Things/KG_Multi_Agent/MKG_Multi_Agent/test1",
                      help='Path to the project directory (default: ./Things/KG_Multi_Agent/MKG_Multi_Agent/test1)')
    parser.add_argument('--exchanges', type=int, default=50,
                      help='Number of last exchanges to extract (default: 50)')

    # Parse arguments
    args = parser.parse_args()

    # Run the main function with the provided project path
    main(args.project_path)
