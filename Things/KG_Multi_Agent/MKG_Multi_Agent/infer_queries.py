# !pip install ollama
# !pip install tqdm


import os
import json
from ollama import Client
import threading
import subprocess
import time
import sys
import argparse
from tqdm import tqdm
import re
from typing import Dict, List, Optional

# Default configuration variables
DEFAULT_PROJECT_PATH = "./Things/KG_Multi_Agent/MKG_Multi_Agent/test1"
DEFAULT_MODEL_NAME = "llama3.2"   #"deepseek-llm:7b"

# The inference query template
inference_query = """###SYSTEM INSTRUCTION###
You are a precise information extraction system. Your task is to analyze conversations and extract EXACTLY 5 structured information requests or hypotheses. You MUST follow the format EXACTLY as shown.

###FORMAT REQUIREMENTS###
- Output EXACTLY 5 requests numbered 1-5
- Include ALL required sections in order: Agents, Tags, Intent, Hypothesis, Rationale, Impact
- Use ONLY [[double_bracket]] format for ALL agents and tags
- Write each section as a SINGLE paragraph (no line breaks)
- Separate requests with "---" on its own line
- Use ONLY alphanumeric characters and [ ] _

###CONCRETE EXAMPLE###
REQUEST 1:
Agents: [[oskar]], [[saffir]], [[ysolda]]
Tags: [[active_inference]], [[knowledge_graph]], [[collaboration]]
Intent: [[research]]

Hypothesis:
The integration of active inference principles into the knowledge graph system will enhance collaborative learning and information sharing among team members.

Rationale:
Understanding how active inference frameworks can be applied to knowledge management will help optimize the way agents interact with and contribute to the shared knowledge base.

Impact:
This research will enable agents to more effectively update their beliefs and make informed decisions based on the collective knowledge available in the system.

---

###VALIDATION RULES###
1. MUST have exactly 5 numbered requests (REQUEST 1: through REQUEST 5:)
2. ALL sections MUST be present and in order for each request
3. ALL agents and tags MUST use [[double_bracket]] format
4. Each section MUST be single paragraph
5. MUST separate requests with "---" on its own line
6. Intent MUST be one of: [[summary]], [[research]], or [[hypothesis_test]]
7. NO special characters except [ ] _ and alphanumeric
8. NO made-up or inferred agents - use ONLY those mentioned in conversation
9. NO explanatory text or additional formatting

###CONVERSATION TO ANALYZE###
"""

def get_ollama_path():
    """Get the path to the Ollama executable based on the operating system."""
    if sys.platform == "win32":
        return os.path.expandvars("%LOCALAPPDATA%\\Programs\\Ollama\\ollama.exe")
    elif sys.platform == "darwin":  # macOS
        return "/usr/local/bin/ollama"
    else:  # Linux and others
        return "/usr/bin/ollama"

def run_ollama_command(command, timeout=None, show_progress=False):
    """Run an Ollama command with proper error handling and optional progress bar."""
    ollama_path = get_ollama_path()
    if not os.path.exists(ollama_path):
        raise FileNotFoundError(f"Ollama executable not found at {ollama_path}")
    
    try:
        if show_progress:
            # For progress tracking, we need to run the command differently
            process = subprocess.Popen(
                [ollama_path] + command.split(),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,
                universal_newlines=True
            )
            
            # Initialize progress bar
            pbar = tqdm(total=100, desc="Downloading model", unit="%")
            last_progress = 0
            
            # Read output line by line
            while True:
                line = process.stderr.readline()
                if not line and process.poll() is not None:
                    break
                    
                # Look for progress information in the output
                if "download:" in line.lower():
                    try:
                        # Extract progress percentage
                        progress_match = re.search(r'(\d+)%', line)
                        if progress_match:
                            current_progress = int(progress_match.group(1))
                            # Update progress bar with the difference
                            pbar.update(current_progress - last_progress)
                            last_progress = current_progress
                    except Exception:
                        pass
            
            pbar.close()
            
            if process.returncode != 0:
                raise subprocess.CalledProcessError(process.returncode, command)
            
            return ""
        else:
            # Regular command execution without progress bar
            result = subprocess.run(
                [ollama_path] + command.split(),
                capture_output=True,
                text=True,
                timeout=timeout
            )
            if result.returncode != 0:
                raise subprocess.CalledProcessError(
                    result.returncode, 
                    command, 
                    result.stdout, 
                    result.stderr
                )
            return result.stdout
            
    except subprocess.TimeoutExpired:
        raise TimeoutError(f"Command '{command}' timed out")
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"Command '{command}' failed: {e.stderr}")

def ensure_model_available(model_name: str):
    """Ensure the specified model is available, downloading it if necessary."""
    print(f"Checking availability of model: {model_name}")
    
    try:
        # Try to list models to check if the model exists
        models_output = run_ollama_command("list")
        if model_name in models_output:
            print(f"Model {model_name} is already available")
            return True
        
        print(f"Model {model_name} not found. Downloading...")
        # Pull the model if not found, showing progress bar
        run_ollama_command(f"pull {model_name}", show_progress=True)
        print(f"\nSuccessfully downloaded model {model_name}")
        return True
        
    except Exception as e:
        print(f"Error ensuring model availability: {str(e)}")
        return False

def start_ollama_server():
    """Start the Ollama server in a separate thread."""
    def run_ollama_serve():
        try:
            run_ollama_command("serve")
        except Exception as e:
            print(f"Error starting Ollama server: {str(e)}")

    thread = threading.Thread(target=run_ollama_serve, daemon=True)
    thread.start()
    time.sleep(5)  # Wait for server to start

def validate_request_format(request_text: str) -> Optional[Dict]:
    """
    Validate a single request format and return structured data if valid.
    Returns None if invalid.
    """
    required_sections = ['Agents:', 'Tags:', 'Intent:', 'Hypothesis:', 'Rationale:', 'Impact:']
    
    # Check if all required sections are present
    for section in required_sections:
        if section not in request_text:
            print(f"Missing required section: {section}")
            return None
    
    # Validate Obsidian links format
    agents_match = re.search(r'Agents:(.*?)(?=Tags:|$)', request_text, re.DOTALL)
    tags_match = re.search(r'Tags:(.*?)(?=Intent:|$)', request_text, re.DOTALL)
    intent_match = re.search(r'Intent:(.*?)(?=Hypothesis:|$)', request_text, re.DOTALL)
    
    if not all([agents_match, tags_match, intent_match]):
        print("Missing or malformed metadata sections")
        return None
    
    # Validate that agents and tags use proper [[link]] format
    agents = agents_match.group(1).strip()
    tags = tags_match.group(1).strip()
    intent = intent_match.group(1).strip()
    
    if not re.search(r'\[\[.*?\]\]', agents) or not re.search(r'\[\[.*?\]\]', tags):
        print("Missing proper [[link]] format in agents or tags")
        return None
    
    # Validate intent is one of the allowed types
    allowed_intents = ['[[summary]]', '[[research]]', '[[hypothesis_test]]']
    if not any(allowed_intent in intent for allowed_intent in allowed_intents):
        print(f"Invalid intent type. Must be one of: {', '.join(allowed_intents)}")
        return None
    
    return {
        'agents': agents,
        'tags': tags,
        'intent': intent,
        'hypothesis': re.search(r'Hypothesis:(.*?)(?=Rationale:|$)', request_text, re.DOTALL).group(1).strip(),
        'rationale': re.search(r'Rationale:(.*?)(?=Impact:|$)', request_text, re.DOTALL).group(1).strip(),
        'impact': re.search(r'Impact:(.*?)(?=---|$)', request_text, re.DOTALL).group(1).strip()
    }

def validate_llm_response(response: str) -> Optional[List[Dict]]:
    """
    Validate the complete LLM response and return structured data if valid.
    Returns None if invalid.
    """
    # Split response into individual requests
    requests = re.split(r'REQUEST \d+:', response)[1:]  # Skip empty first split
    
    # Check if we have exactly 5 requests
    if len(requests) != 5:
        print(f"Invalid number of requests: {len(requests)}. Expected exactly 5.")
        return None
    
    # Validate each request
    validated_requests = []
    for i, request in enumerate(requests, 1):
        print(f"Validating request {i}...")
        validated = validate_request_format(request)
        if not validated:
            print(f"Validation failed for request {i}")
            return None
        validated_requests.append(validated)
    
    return validated_requests

def process_conversation(conversation_text: str, client: Client, model_name: str) -> str:
    """Process a single conversation using the Ollama LLM."""
    # Combine the inference query with the conversation text
    full_prompt = f"{inference_query}\n\nConversation:\n{conversation_text}"
    
    try:
        # Get response from Ollama
        response = client.generate(model=model_name, prompt=full_prompt)
        llm_response = response['response']
        
        # Validate the response
        validated_response = validate_llm_response(llm_response)
        if not validated_response:
            print("Response validation failed. Retrying with more explicit instructions...")
            # Could add retry logic here if needed
            return ""
        
        # Convert validated response back to string format
        formatted_response = json.dumps(validated_response, indent=2)
        return formatted_response
        
    except Exception as e:
        print(f"Error processing conversation: {str(e)}")
        return ""

def main(project_path: str, model_name: str):
    try:
        # Start Ollama server
        start_ollama_server()
        
        # Ensure model is available
        if not ensure_model_available(model_name):
            print("Failed to ensure model availability. Exiting.")
            return
        
        # Initialize Ollama client
        client = Client()
        
        # Define input and output paths based on project path
        input_file = os.path.join(project_path, "outputs", "conversations", "conversations_list.json")
        output_dir = os.path.join(project_path, "outputs", "hypotheses")
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Read conversations
        with open(input_file, 'r', encoding='utf-8') as f:
            conversations = json.load(f)
        
        # Process each conversation and store results
        results = {}
        for conv_name, conv_text in conversations.items():
            print(f"Processing conversation: {conv_name}")
            inferred_queries = process_conversation(conv_text, client, model_name)
            if inferred_queries:  # Only store if we got a valid response
                results[conv_name] = inferred_queries
        
        if not results:
            print("No results were generated. Check the errors above.")
            return
            
        # Save results
        output_file = os.path.join(output_dir, "inferred_queries.json")
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=4, ensure_ascii=False)
        
        print(f"Results saved to: {output_file}")
        
    except Exception as e:
        print(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Process conversations using Ollama LLM to infer queries.')
    parser.add_argument('--project_path', type=str, default=DEFAULT_PROJECT_PATH,
                      help=f'Path to the project directory (default: {DEFAULT_PROJECT_PATH})')
    parser.add_argument('--model_name', type=str, default=DEFAULT_MODEL_NAME,
                      help=f'Name of the Ollama model to use (default: {DEFAULT_MODEL_NAME})')

    # Parse arguments
    args = parser.parse_args()

    # Run the main function with the provided arguments
    main(args.project_path, args.model_name)