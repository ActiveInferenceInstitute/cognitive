import os
import json
import re
import hashlib
import yaml
from datetime import datetime
from typing import Dict, List, Optional
from ollama import Client
import threading
import subprocess
import time
import sys
import argparse
from tqdm import tqdm

# Default configuration
DEFAULT_PROJECT_PATH = "./Things/KG_Multi_Agent/MKG_Multi_Agent/test1"
DEFAULT_MODEL_NAME = "llama3.2"

# Reuse the inference query template from infer_queries.py
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

---"""

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
            process = subprocess.Popen(
                [ollama_path] + command.split(),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,
                universal_newlines=True
            )
            
            pbar = tqdm(total=100, desc="Downloading model", unit="%")
            last_progress = 0
            
            while True:
                line = process.stderr.readline()
                if not line and process.poll() is not None:
                    break
                    
                if "download:" in line.lower():
                    try:
                        progress_match = re.search(r'(\d+)%', line)
                        if progress_match:
                            current_progress = int(progress_match.group(1))
                            pbar.update(current_progress - last_progress)
                            last_progress = current_progress
                    except Exception:
                        pass
            
            pbar.close()
            
            if process.returncode != 0:
                raise subprocess.CalledProcessError(process.returncode, command)
            
            return ""
        else:
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
        models_output = run_ollama_command("list")
        if model_name in models_output:
            print(f"Model {model_name} is already available")
            return True
        
        print(f"Model {model_name} not found. Downloading...")
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

def format_request_for_obsidian(request: Dict, source_info: Dict) -> Dict:
    """Format a research request into Obsidian-compatible markdown structure."""
    # Generate a unique ID for the request based on content
    content_hash = hashlib.md5(
        f"{request['hypothesis']}{request['rationale']}".encode()
    ).hexdigest()[:8]
    request_id = f"request_{content_hash}"
    
    # Create frontmatter metadata
    frontmatter = {
        "source_conversation": source_info["conversation_file"],
        "source_chunk": source_info.get("chunk_id", "full"),
        "type": request["intent"].strip("[]"),
        "created": datetime.now().strftime("%Y-%m-%d"),
        "tags": [tag.strip("[]") for tag in request["tags"].split(",")],
        "agents": [agent.strip("[]") for agent in request["agents"].split(",")]
    }
    
    # Create markdown content
    markdown_content = f"""---
{yaml.dump(frontmatter)}---

# Research Request: {request['hypothesis']}

## Context and Rationale
{request['rationale']}

## Expected Impact
{request['impact']}

## Related Agents
{", ".join([f"[[{agent}]]" for agent in frontmatter["agents"]])}

## Tags
{", ".join([f"[[{tag}]]" for tag in frontmatter["tags"]])}

## Source Reference
[[{source_info['conversation_file']}]]
"""
    
    return {
        "id": request_id,
        "content": markdown_content,
        "metadata": frontmatter
    }

def validate_request_format(request_text: str) -> Optional[Dict]:
    """Validate a single request format and return structured data if valid."""
    required_sections = ['Agents:', 'Tags:', 'Intent:', 'Hypothesis:', 'Rationale:', 'Impact:']
    
    # Check if all required sections are present
    for section in required_sections:
        if section not in request_text:
            print(f"Missing required section: {section}")
            return None
    
    # Extract sections using regex
    agents_match = re.search(r'Agents:(.*?)(?=Tags:|$)', request_text, re.DOTALL)
    tags_match = re.search(r'Tags:(.*?)(?=Intent:|$)', request_text, re.DOTALL)
    intent_match = re.search(r'Intent:(.*?)(?=Hypothesis:|$)', request_text, re.DOTALL)
    hypothesis_match = re.search(r'Hypothesis:(.*?)(?=Rationale:|$)', request_text, re.DOTALL)
    rationale_match = re.search(r'Rationale:(.*?)(?=Impact:|$)', request_text, re.DOTALL)
    impact_match = re.search(r'Impact:(.*?)(?=---|$)', request_text, re.DOTALL)
    
    if not all([agents_match, tags_match, intent_match, hypothesis_match, rationale_match, impact_match]):
        print("Failed to extract all required sections")
        return None
    
    # Validate [[link]] format
    agents = agents_match.group(1).strip()
    tags = tags_match.group(1).strip()
    intent = intent_match.group(1).strip()
    
    if not all(re.findall(r'\[\[.*?\]\]', section) for section in [agents, tags, intent]):
        print("Missing proper [[link]] format in agents, tags, or intent")
        return None
    
    # Validate intent type
    allowed_intents = ['[[summary]]', '[[research]]', '[[hypothesis_test]]']
    if not any(allowed_intent in intent for allowed_intent in allowed_intents):
        print(f"Invalid intent type. Must be one of: {', '.join(allowed_intents)}")
        return None
    
    return {
        'agents': agents,
        'tags': tags,
        'intent': intent,
        'hypothesis': hypothesis_match.group(1).strip(),
        'rationale': rationale_match.group(1).strip(),
        'impact': impact_match.group(1).strip()
    }

def validate_llm_response(response: str) -> Optional[List[Dict]]:
    """Validate the complete LLM response and return structured data if valid."""
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

def process_conversation(conversation_text: str, client: Client, model_name: str, source_info: Dict) -> List[Dict]:
    """Process a single conversation using the Ollama LLM."""
    # Combine the inference query with the conversation text
    full_prompt = f"{inference_query}\n\nConversation:\n{conversation_text}"
    
    try:
        # Get response from Ollama
        response = client.generate(model=model_name, prompt=full_prompt)
        llm_response = response['response']
        
        # Validate the response
        validated_requests = validate_llm_response(llm_response)
        if not validated_requests:
            print("Response validation failed")
            return []
        
        # Format requests for Obsidian
        obsidian_requests = []
        for request in validated_requests:
            formatted = format_request_for_obsidian(request, source_info)
            obsidian_requests.append(formatted)
        
        return obsidian_requests
        
    except Exception as e:
        print(f"Error processing conversation: {str(e)}")
        return []

def save_obsidian_files(requests: List[Dict], output_dir: str):
    """Save formatted requests as individual Obsidian markdown files."""
    vault_dir = os.path.join(output_dir, "vault")
    os.makedirs(vault_dir, exist_ok=True)
    
    # Create index for tracking
    request_index = {}
    
    for request in requests:
        # Save markdown file
        filename = f"{request['id']}.md"
        filepath = os.path.join(vault_dir, filename)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(request['content'])
        
        # Update index
        request_index[request['id']] = {
            'title': request['metadata']['type'],
            'tags': request['metadata']['tags'],
            'agents': request['metadata']['agents'],
            'source': request['metadata']['source_conversation']
        }
    
    # Save index file
    index_path = os.path.join(vault_dir, "_request_index.json")
    with open(index_path, 'w', encoding='utf-8') as f:
        json.dump(request_index, f, indent=2)

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
        
        # Define input and output paths
        input_file = os.path.join(project_path, "outputs", "conversations", "conversations_list.json")
        output_dir = os.path.join(project_path, "outputs", "research_requests")
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Read conversations
        with open(input_file, 'r', encoding='utf-8') as f:
            conversations = json.load(f)
        
        # Process each conversation
        all_requests = []
        for conv_name, conv_text in conversations.items():
            print(f"Processing conversation: {conv_name}")
            source_info = {
                "conversation_file": conv_name,
                "chunk_id": "full"  # Could be updated if implementing chunking
            }
            
            requests = process_conversation(conv_text, client, model_name, source_info)
            all_requests.extend(requests)
        
        if not all_requests:
            print("No valid requests were generated. Check the errors above.")
            return
        
        # Save as Obsidian vault files
        save_obsidian_files(all_requests, output_dir)
        print(f"Successfully processed {len(all_requests)} requests into Obsidian vault format")
        
    except Exception as e:
        print(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process conversations into Obsidian vault research requests.')
    parser.add_argument('--project_path', type=str, default=DEFAULT_PROJECT_PATH,
                      help=f'Path to the project directory (default: {DEFAULT_PROJECT_PATH})')
    parser.add_argument('--model_name', type=str, default=DEFAULT_MODEL_NAME,
                      help=f'Name of the Ollama model to use (default: {DEFAULT_MODEL_NAME})')

    args = parser.parse_args()
    main(args.project_path, args.model_name) 