# !pip install ollama
# install ollama from https://ollama.ai/download
# !pip install tqdm
# !pip install pyyaml
# !pip install plotly
# !pip install "plotly[express]" networkx

import os
import json
import re
import hashlib
import yaml
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any
from ollama import Client
import threading
import subprocess
import time
import sys
import argparse
from tqdm import tqdm
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import networkx as nx
from collections import Counter, defaultdict
import pandas as pd

# Default configuration
DEFAULT_PROJECT_PATH = "./Things/KG_Multi_Agent/MKG_Multi_Agent/test1"
DEFAULT_MODEL_NAME = "llama3.2"  # Choose one model name

# Reuse the inference query template from infer_queries.py
inference_query = """###SYSTEM INSTRUCTION###
You are a precise information extraction system. Your task is to analyze the conversation and extract ONE research request or hypothesis. Format your response as a single JSON object.

###FORMAT REQUIREMENTS###
- Extract exactly ONE research request
- Include ALL required fields: agents, tags, intent, hypothesis, rationale, impact
- Use ONLY [[double_bracket]] format for agents, tags, and intent
- Each field should be a single string (no line breaks)
- Use only alphanumeric characters and [ ] _

###OUTPUT FORMAT###
{
    "agents": "[[agent1]], [[agent2]]",
    "tags": "[[tag1]], [[tag2]], [[tag3]]",
    "intent": "[[research]]",
    "hypothesis": "Clear detailed hypothesis statement",
    "rationale": "Clear concise rationale explanation",
    "impact": "Clear concise impact statement"
}

###CONVERSATION CHUNK###
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
    """Start the Ollama server in a separate thread if not already running."""
    # Try to connect to Ollama server first to check if it's running
    try:
        client = Client(host='http://127.0.0.1:11434')
        client.list()  # This will fail if server is not running
        print("Ollama server is already running")
        return
    except Exception:
        print("Starting Ollama server...")
        def run_ollama_serve():
            try:
                run_ollama_command("serve")
            except Exception as e:
                print(f"Error starting Ollama server: {str(e)}")

        thread = threading.Thread(target=run_ollama_serve, daemon=True)
        thread.start()
        time.sleep(5)  #

def format_request_for_obsidian(request: Dict, source_info: Dict) -> Dict:
    """Format a research request into Obsidian-compatible markdown structure."""
    # Generate timestamp
    timestamp = datetime.now()
    formatted_timestamp = timestamp.strftime("%Y%m%d_%H%M%S")
    
    # Generate a unique ID for the request
    content_hash = hashlib.md5(
        f"{request['hypothesis']}{request['rationale']}{formatted_timestamp}".encode()
    ).hexdigest()[:8]
    request_id = f"request_{formatted_timestamp}_{content_hash}"
    
    # Clean up agents and tags
    cleaned_agents = [
        fix_bracket_links(agent.strip()) 
        for agent in request['agents'].split(',')
        if agent.strip()
    ]
    cleaned_tags = [
        fix_bracket_links(tag.strip())
        for tag in request['tags'].split(',')
        if tag.strip()
    ]
    
    # Create comprehensive frontmatter with all fields
    frontmatter = {
        # Source information
        "source_conversation": source_info["conversation_file"],
        "source_chunk": source_info.get("chunk_id", "full"),
        
        # Core fields from request
        "type": request["intent"].strip("[]"),
        "hypothesis": request["hypothesis"],
        "rationale": request["rationale"],
        "impact": request["impact"],
        
        # Cleaned arrays
        "tags": [tag.strip("[]") for tag in cleaned_tags],
        "agents": [agent.strip("[]") for agent in cleaned_agents],
        
        # Timestamps
        "created": timestamp.strftime("%Y-%m-%d"),
        "timestamp": timestamp.strftime("%Y-%m-%d %H:%M:%S"),
    }
    
    # Create markdown content with proper link formatting
    markdown_content = f"""---
{yaml.dump(frontmatter)}---

# Research Request: {request['hypothesis']}

## Created
{timestamp.strftime("%Y-%m-%d %H:%M:%S")}

## Hypothesis
{request['hypothesis']}

## Context and Rationale
{request['rationale']}

## Expected Impact
{request['impact']}

## Related Agents
{', '.join(cleaned_agents)}

## Tags
{', '.join(cleaned_tags)}

## Source Reference
[[{source_info['conversation_file']}]]
"""
    
    return {
        "id": request_id,
        "content": markdown_content,
        "metadata": frontmatter,
        "timestamp": timestamp
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

def chunk_conversation(conversation_text: str, max_chunk_size: int = 4000) -> List[str]:
    """Split conversation into manageable chunks while preserving context."""
    # Split on message boundaries (e.g. speaker changes)
    messages = re.split(r'(\w+: )', conversation_text)
    chunks = []
    current_chunk = ""
    
    for i in range(0, len(messages)-1, 2):
        speaker = messages[i+1]
        content = messages[i+2] if i+2 < len(messages) else ""
        message = speaker + content
        
        if len(current_chunk) + len(message) > max_chunk_size:
            if current_chunk:
                chunks.append(current_chunk)
            current_chunk = message
        else:
            current_chunk += message
            
    if current_chunk:
        chunks.append(current_chunk)
        
    return chunks

def fix_bracket_links(text: str) -> str:
    """
    Fix malformed bracket links in text by ensuring proper closure of [[ ]] pairs.
    Rules:
    1. When [[ is found, look for next ]] or [[
    2. If ]] found first, continue to next [[
    3. If [[ found first, close previous [[ with ]] before word boundary
    4. Convert single brackets [text] to double brackets [[text]]
    """
    # First, convert single brackets to double brackets
    # But we need to be careful not to affect existing double brackets
    i = 0
    result = ""
    while i < len(text):
        if i + 1 < len(text) and text[i:i+2] == "[[":
            # Skip over existing double-bracketed content
            next_close = text.find("]]", i+2)
            if next_close != -1:
                result += text[i:next_close+2]
                i = next_close + 2
                continue
        
        if text[i] == "[" and (i == 0 or text[i-1:i+1] != "[["):
            # Found a single opening bracket
            # Look for matching single closing bracket
            j = i + 1
            while j < len(text):
                if text[j] == "]" and (j+1 >= len(text) or text[j:j+2] != "]]"):
                    # Found matching single closing bracket
                    # Convert [text] to [[text]]
                    content = text[i+1:j]
                    result += "[[" + content + "]]"
                    i = j + 1
                    break
                j += 1
            else:
                # No matching closing bracket found
                result += text[i]
                i += 1
        else:
            result += text[i]
            i += 1
    
    # Now handle any remaining malformed double brackets
    text = result
    result = ""
    i = 0
    while i < len(text):
        # Look for opening brackets
        if i + 1 < len(text) and text[i:i+2] == "[[":
            # Find the start of the word after [[
            word_start = i + 2
            # Find the next [[ or ]]
            next_open = text.find("[[", word_start)
            next_close = text.find("]]", word_start)
            
            # If no more closing brackets found, add ]] at next word boundary
            if next_close == -1:
                # Find next word boundary (space, punctuation, etc.)
                j = word_start
                while j < len(text) and text[j].isalnum():
                    j += 1
                result += text[i:j] + "]]"
                i = j
                continue
                
            # If next_open comes before next_close or no close found, 
            # close the current word
            if next_open != -1 and (next_close == -1 or next_open < next_close):
                # Find the end of current word
                j = word_start
                while j < next_open and text[j].isalnum():
                    j += 1
                result += text[i:j] + "]]"
                i = j
            else:
                # Normal case: [[ ... ]] found
                result += text[i:next_close + 2]
                i = next_close + 2
        else:
            result += text[i]
            i += 1
    
    return result

def extract_json_from_response(response: str) -> Optional[Dict]:
    """Extract and validate JSON from LLM response with cleanup."""
    try:
        # Find JSON-like content
        json_match = re.search(r'\{[\s\S]*\}', response)
        if not json_match:
            return None
            
        json_str = json_match.group(0)
        
        # Fix malformed bracket links before cleaning
        json_str = fix_bracket_links(json_str)
        
        # Clean up common JSON formatting issues
        json_str = re.sub(r'[\n\r]', ' ', json_str)
        json_str = re.sub(r'\s+', ' ', json_str)
        
        # Pretty print the fixed JSON for debugging
        print("\n🔍 Fixed JSON string:")
        print("-------------------")
        print(json_str)
        print("-------------------\n")
        
        data = json.loads(json_str)
        
        # Additional validation for bracket links in specific fields
        for field in ['agents', 'tags', 'intent']:
            if field in data:
                data[field] = fix_bracket_links(data[field])
        
        return data
    except Exception as e:
        print(f"⚠️ Error processing JSON: {str(e)}")
        return None

def validate_research_request(request: Dict) -> bool:
    """Validate research request format and content."""
    required_fields = ['agents', 'tags', 'intent', 'hypothesis', 'rationale', 'impact']
    
    # Check all fields exist
    if not all(field in request for field in required_fields):
        print("❌ Missing required fields")
        return False
        
    # Validate [[brackets]] format
    for field in ['agents', 'tags', 'intent']:
        values = [v.strip() for v in request[field].split(',')]
        for value in values:
            if not (value.startswith('[[') and value.endswith(']]')):
                print(f"❌ Invalid bracket format in {field}: {value}")
                return False
            # Check for nested brackets
            if value.count('[[') > 1 or value.count(']]') > 1:
                print(f"❌ Nested brackets found in {field}: {value}")
                return False
            
    # Validate content exists
    if not all(request[field].strip() for field in required_fields):
        print("❌ Empty field found")
        return False
        
    return True

def process_conversation(conversation_text: str, client: Client, model_name: str, source_info: Dict) -> List[Dict]:
    """Process conversation chunks and extract research requests."""
    chunks = chunk_conversation(conversation_text)
    requests = []
    
    # Create progress bar for chunks
    with tqdm(total=len(chunks), desc="Processing conversation chunks") as pbar:
        for i, chunk in enumerate(chunks):
            chunk_info = {
                **source_info,
                "chunk_id": f"chunk_{i+1}"
            }
            
            # Combine inference query with chunk
            prompt = f"{inference_query}\n\n{chunk}"
            
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    print(f"\n{'='*80}\nProcessing chunk {i+1}/{len(chunks)} (Attempt {attempt+1})\n{'='*80}")
                    
                    response = client.generate(model=model_name, prompt=prompt)
                    request = extract_json_from_response(response['response'])
                    
                    if request and validate_research_request(request):
                        formatted = format_request_for_obsidian(request, chunk_info)
                        requests.append(formatted)
                        
                        # Pretty print the extracted request
                        print("\n📋 Extracted Research Request:")
                        print("----------------------------")
                        print(json.dumps(request, indent=2))
                        print("----------------------------\n")
                        break
                    else:
                        print("❌ Failed to validate request format")
                        
                except Exception as e:
                    print(f"⚠️ Error processing chunk {i+1}, attempt {attempt+1}: {str(e)}")
                    if attempt == max_retries - 1:
                        print(f"❌ Failed to process chunk {i+1} after {max_retries} attempts")
            
            pbar.update(1)
            
    return requests

def merge_request_index(existing_index: Dict, new_requests: List[Dict], output_dir: str) -> Dict:
    """Merge new requests into existing request index."""
    merged_index = existing_index.copy() if existing_index else {}
    
    # Add new requests to the index
    for request in new_requests:
        request_id = request['id']
        if request_id not in merged_index:  # Only add if not already present
            merged_index[request_id] = {
                'title': request['metadata']['type'],
                'tags': request['metadata']['tags'],
                'agents': request['metadata']['agents'],
                'source': request['metadata']['source_conversation'],
                'timestamp': request['metadata']['timestamp']
            }
    
    return merged_index

def load_existing_requests(output_dir: str) -> Tuple[Dict, List[Dict]]:
    """Load existing request index and convert to request objects."""
    index_path = os.path.join(output_dir, "_request_index.json")
    existing_requests = []
    existing_index = {}
    
    if os.path.exists(index_path):
        with open(index_path, 'r', encoding='utf-8') as f:
            existing_index = json.load(f)
            
        # Convert index entries back to request objects
        for request_id, metadata in existing_index.items():
            request_obj = {
                'id': request_id,
                'metadata': {
                    'type': metadata['title'],
                    'tags': metadata['tags'],
                    'agents': metadata['agents'],
                    'source_conversation': metadata['source'],
                    'timestamp': metadata['timestamp']
                },
                'timestamp': datetime.strptime(metadata['timestamp'], "%Y-%m-%d %H:%M:%S")
            }
            existing_requests.append(request_obj)
    
    return existing_index, existing_requests

def save_obsidian_files(requests: List[Dict], output_dir: str):
    """Save formatted requests as individual Obsidian markdown files and update indexes."""
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Load existing data
    existing_index, existing_requests = load_existing_requests(output_dir)
    
    # Add progress bar for saving files
    with tqdm(total=len(requests), desc="Saving research requests") as pbar:
        for request in requests:
            # Save markdown file
            filename = f"{request['id']}.md"
            filepath = os.path.join(output_dir, filename)
            
            # Only write if file doesn't exist (preserve existing files)
            if not os.path.exists(filepath):
                with open(filepath, 'w', encoding='utf-8') as f:
                    f.write(request['content'])
                
                # Print saved file info
                print(f"\n📝 Saved research request: {filename}")
                print(f"   Type: {request['metadata']['type']}")
                print(f"   Tags: {', '.join(request['metadata']['tags'])}")
                print(f"   Agents: {', '.join(request['metadata']['agents'])}")
                print(f"   Created: {request['metadata']['timestamp']}\n")
            
            pbar.update(1)
    
    # Merge and save JSON index
    merged_index = merge_request_index(existing_index, requests, output_dir)
    index_path = os.path.join(output_dir, "_request_index.json")
    with open(index_path, 'w', encoding='utf-8') as f:
        json.dump(merged_index, f, indent=2)
    print(f"📚 Updated JSON index: {index_path}")
    
    # Combine existing and new requests for the markdown index
    all_requests = existing_requests + requests
    
    # Create and save markdown index
    create_research_index(all_requests, output_dir)
    print(f"📚 Updated markdown index: {os.path.join(output_dir, '_research_requests_index.md')}")

def create_research_index(requests: List[Dict], output_dir: str):
    """Create or update an Obsidian-compatible index file linking all research requests."""
    print(f"\nUpdating index with {len(requests)} total requests")
    
    # Remove any duplicate requests based on ID
    seen_ids = set()
    unique_requests = []
    for request in requests:
        if request['id'] not in seen_ids:
            seen_ids.add(request['id'])
            unique_requests.append(request)
    
    # Calculate statistics
    total_requests = len(unique_requests)
    
    # Agent statistics
    agent_counts = {}
    for request in unique_requests:
        for agent in request['metadata']['agents']:
            agent_counts[agent] = agent_counts.get(agent, 0) + 1
    
    # Tag statistics
    tag_counts = {}
    for request in unique_requests:
        for tag in request['metadata']['tags']:
            tag_counts[tag] = tag_counts.get(tag, 0) + 1
    
    # Create index content with statistics sections
    index_content = """# Research Requests Index

## Statistics Summary

### Agent Statistics
| Agent | Contributions | Percentage |
|-------|--------------|------------|
"""
    
    # Add agent statistics
    for agent, count in sorted(agent_counts.items(), key=lambda x: x[1], reverse=True):
        percentage = (count / total_requests) * 100
        # Ensure agent is enclosed in [[]] if not already
        linked_agent = f"[[{agent.strip('[]')}]]"
        index_content += f"|{linked_agent}|{count}/{total_requests}|{percentage:.1f}%|\n"
    
    index_content += "\n### Tag Statistics\n"
    index_content += "| Tag | Usage | Percentage |\n"
    index_content += "|-----|-------|------------|\n"
    
    # Add tag statistics
    for tag, count in sorted(tag_counts.items(), key=lambda x: x[1], reverse=True):
        percentage = (count / total_requests) * 100
        # Ensure tag is enclosed in [[]] if not already
        linked_tag = f"[[{tag.strip('[]')}]]"
        index_content += f"|{linked_tag}|{count}/{total_requests}|{percentage:.1f}%|\n"
    
    # Continue with existing sections
    index_content += "\n## By Date\n"
    
    # Sort all requests by timestamp
    sorted_requests = sorted(unique_requests, key=lambda x: x['timestamp'])
    
    # Group by date
    date_groups = {}
    for request in sorted_requests:
        date = request['timestamp'].strftime("%Y-%m-%d")
        if date not in date_groups:
            date_groups[date] = []
        date_groups[date].append(request)
    
    # Add date sections
    for date in sorted(date_groups.keys()):
        date_requests = date_groups[date]
        index_content += f"\n### {date}\n\n"
        for request in sorted(date_requests, key=lambda x: x['timestamp']):
            metadata = request['metadata']
            hypothesis_summary = metadata.get('hypothesis', '')[:100] + "..." if metadata.get('hypothesis', '') else ''
            type_str = metadata.get('type', '').capitalize()
            
            index_content += f"- [[{request['id']}]] - {type_str}: {hypothesis_summary}\n"
    
    # Add agent sections
    index_content += "\n## By Agent\n"
    agent_groups = {}
    for request in unique_requests:
        for agent in request['metadata']['agents']:
            if agent not in agent_groups:
                agent_groups[agent] = []
            agent_groups[agent].append(request)
    
    for agent in sorted(agent_groups.keys()):
        index_content += f"\n### [[{agent}]]\n\n"
        for request in sorted(agent_groups[agent], key=lambda x: x['timestamp'], reverse=True):
            hypothesis_summary = request['metadata'].get('hypothesis', '')[:50] + "..." if request['metadata'].get('hypothesis', '') else ''
            index_content += f"- [[{request['id']}]] - {hypothesis_summary} ({request['timestamp'].strftime('%Y-%m-%d %H:%M:%S')})\n"
    
    # Add tag sections
    index_content += "\n## By Tag\n"
    tag_groups = {}
    for request in unique_requests:
        for tag in request['metadata']['tags']:
            if tag not in tag_groups:
                tag_groups[tag] = []
            tag_groups[tag].append(request)
    
    for tag in sorted(tag_groups.keys()):
        index_content += f"\n### [[{tag}]]\n\n"
        for request in sorted(tag_groups[tag], key=lambda x: x['timestamp'], reverse=True):
            hypothesis_summary = request['metadata'].get('hypothesis', '')[:50] + "..." if request['metadata'].get('hypothesis', '') else ''
            index_content += f"- [[{request['id']}]] - {hypothesis_summary} ({request['timestamp'].strftime('%Y-%m-%d %H:%M:%S')})\n"
    
    # Save index file
    index_path = os.path.join(output_dir, "_research_requests_index.md")
    with open(index_path, 'w', encoding='utf-8') as f:
        f.write(index_content)
    
    print(f"📚 Updated research requests index: {index_path}")

    # Add visualization generation
    create_statistics_visualizations(requests, output_dir)
    create_network_visualizations(requests, output_dir)

def create_statistics_visualizations(requests_data: List[Dict], output_dir: str):
    """Create and save various statistical visualizations of the research requests data."""
    # Create visualizations directory if it doesn't exist
    viz_dir = os.path.join(output_dir, "visualizations")
    os.makedirs(viz_dir, exist_ok=True)
    
    # 1. Agent Participation Bar Chart with improved styling
    agent_counts = Counter()
    total_requests = len(requests_data)
    for request in requests_data:
        for agent in request['metadata'].get('agents', []):
            agent_counts[agent] = agent_counts.get(agent, 0) + 1
    
    fig_agents = px.bar(
        x=list(agent_counts.keys()),
        y=[count/total_requests * 100 for count in agent_counts.values()],
        title="Agent Participation in Research Requests",
        labels={'x': 'Agent', 'y': 'Participation Rate (%)'},
        color=[count/total_requests * 100 for count in agent_counts.values()],
        color_continuous_scale='Viridis'
    )
    fig_agents.update_layout(
        plot_bgcolor='white',
        showlegend=False,
        title_x=0.5,
        title_font_size=20
    )
    fig_agents.write_html(os.path.join(viz_dir, "agent_participation.html"))
    
    # 2. Tag Usage Treemap with improved hierarchy
    tag_counts = Counter()
    for request in requests_data:
        for tag in request['metadata'].get('tags', []):
            tag_counts[tag] = tag_counts.get(tag, 0) + 1
    
    # Create hierarchical structure for tags
    tag_data = []
    for tag, count in tag_counts.most_common():
        # Split tag into parts for hierarchy
        parts = tag.strip('[]').split('_')
        current_path = ''
        for i, part in enumerate(parts):
            parent = current_path if i > 0 else 'Tags'
            current_path = part if i == 0 else f"{current_path}_{part}"
            tag_data.append({
                'id': current_path,
                'parent': parent,
                'value': count if i == len(parts) - 1 else None
            })
    
    # Convert to DataFrame for plotly
    df_tags = pd.DataFrame(tag_data)
    
    fig_tags = px.treemap(
        df_tags,
        ids='id',
        parents='parent',
        values='value',
        title="Tag Usage Distribution",
        color='value',
        color_continuous_scale='RdBu'
    )
    fig_tags.update_layout(title_x=0.5, title_font_size=20)
    fig_tags.write_html(os.path.join(viz_dir, "tag_usage.html"))
    
    # 3. Time Series of Request Activity
    dates = [request['timestamp'] for request in requests_data]
    date_counts = Counter(date.date() for date in dates)
    
    fig_timeline = px.line(
        x=sorted(date_counts.keys()),
        y=[date_counts[date] for date in sorted(date_counts.keys())],
        title="Research Request Activity Over Time",
        labels={'x': 'Date', 'y': 'Number of Requests'},
        markers=True
    )
    fig_timeline.update_layout(
        plot_bgcolor='white',
        title_x=0.5,
        title_font_size=20
    )
    fig_timeline.write_html(os.path.join(viz_dir, "request_timeline.html"))
    
    # 4. Agent-Tag Heatmap
    agent_tag_matrix = defaultdict(lambda: defaultdict(int))
    for request in requests_data:
        for agent in request['metadata'].get('agents', []):
            for tag in request['metadata'].get('tags', []):
                agent_tag_matrix[agent][tag] += 1
    
    # Convert to matrix format
    agents = sorted(set(agent for request in requests_data for agent in request['metadata'].get('agents', [])))
    tags = sorted(set(tag for request in requests_data for tag in request['metadata'].get('tags', [])))
    
    heatmap_data = [[agent_tag_matrix[agent][tag] for tag in tags] for agent in agents]
    
    fig_heatmap = go.Figure(data=go.Heatmap(
        z=heatmap_data,
        x=tags,
        y=agents,
        colorscale='Viridis'
    ))
    
    fig_heatmap.update_layout(
        title='Agent-Tag Collaboration Heatmap',
        title_x=0.5,
        title_font_size=20,
        xaxis_title='Tags',
        yaxis_title='Agents',
        xaxis={'tickangle': 45}
    )
    fig_heatmap.write_html(os.path.join(viz_dir, "agent_tag_heatmap.html"))
    
    # 5. Create a Sankey diagram showing flow between agents and tags
    agent_indices = {agent: i for i, agent in enumerate(agents)}
    tag_indices = {tag: i + len(agents) for i, tag in enumerate(tags)}
    
    sankey_data = defaultdict(int)
    for request in requests_data:
        for agent in request['metadata'].get('agents', []):
            for tag in request['metadata'].get('tags', []):
                sankey_data[(agent_indices[agent], tag_indices[tag])] += 1
    
    fig_sankey = go.Figure(data=[go.Sankey(
        node=dict(
            pad=15,
            thickness=20,
            line=dict(color="black", width=0.5),
            label=agents + tags,
            color=["#1f77b4"] * len(agents) + ["#ff7f0e"] * len(tags)
        ),
        link=dict(
            source=[s for s, _ in sankey_data.keys()],
            target=[t for _, t in sankey_data.keys()],
            value=list(sankey_data.values())
        )
    )])
    
    fig_sankey.update_layout(
        title='Agent-Tag Flow Diagram',
        title_x=0.5,
        title_font_size=20,
        font_size=10
    )
    fig_sankey.write_html(os.path.join(viz_dir, "agent_tag_flow.html"))

def create_network_visualizations(requests_data: List[Dict], output_dir: str):
    """Create and save network visualizations of the research requests data."""
    viz_dir = os.path.join(output_dir, "visualizations")
    
    # Create a graph
    G = nx.Graph()
    
    # Add nodes and edges with metadata
    for request in requests_data:
        request_id = request['id']
        
        # Add request node
        G.add_node(request_id, 
                  node_type='request',
                  hypothesis=request['metadata'].get('hypothesis', ''),
                  timestamp=request['timestamp'])
        
        # Add agents and connect to request
        for agent in request['metadata'].get('agents', []):
            if not G.has_node(agent):
                G.add_node(agent, node_type='agent')
            G.add_edge(agent, request_id, edge_type='agent_request')
        
        # Add tags and connect to request
        for tag in request['metadata'].get('tags', []):
            if not G.has_node(tag):
                G.add_node(tag, node_type='tag')
            G.add_edge(tag, request_id, edge_type='tag_request')
    
    # Calculate network metrics
    degree_centrality = nx.degree_centrality(G)
    betweenness_centrality = nx.betweenness_centrality(G)
    
    # Create node traces for different types
    def create_node_trace(nodes, node_type, color):
        x = []
        y = []
        text = []
        sizes = []
        
        pos = nx.spring_layout(G, k=1, iterations=50)
        
        for node in nodes:
            x.append(pos[node][0])
            y.append(pos[node][1])
            
            # Create hover text with metrics
            hover_text = f"{node}<br>"
            hover_text += f"Degree Centrality: {degree_centrality[node]:.3f}<br>"
            hover_text += f"Betweenness Centrality: {betweenness_centrality[node]:.3f}<br>"
            
            if node_type == 'request':
                hover_text += f"Hypothesis: {G.nodes[node]['hypothesis'][:100]}...<br>"
                hover_text += f"Time: {G.nodes[node]['timestamp']}"
            
            text.append(hover_text)
            sizes.append(30 + 70 * degree_centrality[node])
        
        return go.Scatter(
            x=x, y=y,
            mode='markers+text',
            hoverinfo='text',
            text=[node for node in nodes],
            textposition="top center",
            hovertext=text,
            marker=dict(
                color=color,
                size=sizes,
                line=dict(width=2),
                symbol='circle'
            ),
            name=node_type
        )
    
    # Separate nodes by type
    agent_nodes = [n for n, attr in G.nodes(data=True) if attr.get('node_type') == 'agent']
    tag_nodes = [n for n, attr in G.nodes(data=True) if attr.get('node_type') == 'tag']
    request_nodes = [n for n, attr in G.nodes(data=True) if attr.get('node_type') == 'request']
    
    # Create edge traces
    edge_traces = []
    pos = nx.spring_layout(G, k=1, iterations=50)
    
    edge_colors = {
        'agent_request': 'rgba(31, 119, 180, 0.3)',
        'tag_request': 'rgba(255, 127, 14, 0.3)'
    }
    
    for edge_type, color in edge_colors.items():
        edge_x = []
        edge_y = []
        
        for edge in G.edges(data=True):
            if edge[2].get('edge_type') == edge_type:
                x0, y0 = pos[edge[0]]
                x1, y1 = pos[edge[1]]
                edge_x.extend([x0, x1, None])
                edge_y.extend([y0, y1, None])
        
        edge_traces.append(
            go.Scatter(
                x=edge_x, y=edge_y,
                line=dict(width=0.5, color=color),
                hoverinfo='none',
                mode='lines',
                name=edge_type
            )
        )
    
    # Create figure with all traces
    fig = go.Figure(data=[
        *edge_traces,
        create_node_trace(agent_nodes, 'agent', '#1f77b4'),
        create_node_trace(tag_nodes, 'tag', '#ff7f0e'),
        create_node_trace(request_nodes, 'request', '#2ca02c')
    ])
    
    # Update layout
    fig.update_layout(
        title='Research Request Network',
        title_x=0.5,
        title_font_size=20,
        showlegend=True,
        hovermode='closest',
        margin=dict(b=20, l=5, r=5, t=40),
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        plot_bgcolor='white'
    )
    
    # Save network visualization
    fig.write_html(os.path.join(viz_dir, "network_visualization.html"))
    
    # Calculate and save network statistics
    stats = {
        'Number of Nodes': len(G.nodes()),
        'Number of Edges': len(G.edges()),
        'Average Degree': sum(dict(G.degree()).values()) / len(G.nodes()),
        'Network Density': nx.density(G),
        'Average Clustering Coefficient': nx.average_clustering(G),
        'Top Agents by Centrality': sorted(
            [(n, c) for n, c in degree_centrality.items() if n in agent_nodes],
            key=lambda x: x[1],
            reverse=True
        )[:5],
        'Top Tags by Centrality': sorted(
            [(n, c) for n, c in degree_centrality.items() if n in tag_nodes],
            key=lambda x: x[1],
            reverse=True
        )[:5]
    }
    
    # Save network statistics
    with open(os.path.join(viz_dir, 'network_statistics.json'), 'w') as f:
        json.dump(stats, f, indent=2, default=str)

def test_fix_bracket_links():
    """Test the bracket fixing functionality with various cases."""
    test_cases = [
        (
            '{"agents": "[[person1]], [[person2", "tags": "[[tag1]], tag2[["}',
            '{"agents": "[[person1]], [[person2]]", "tags": "[[tag1]], [[tag2]]"}'
        ),
        (
            '[[unclosed [[word]] test',
            '[[unclosed]] [[word]] test'
        ),
        (
            'nested [[bracket [[test]] here',
            'nested [[bracket]] [[test]] here'
        ),
        (
            '[[multiple [[words]] in [[one]] string',
            '[[multiple]] [[words]] in [[one]] string'
        ),
        # New test cases for single brackets
        (
            '[single] bracket test',
            '[[single]] bracket test'
        ),
        (
            'mixed [single] and [[double]] brackets',
            'mixed [[single]] and [[double]] brackets'
        ),
        (
            '[multiple] [single] brackets',
            '[[multiple]] [[single]] brackets'
        ),
        (
            'nested [outer [inner] test]',
            'nested [[outer inner test]]'
        ),
        (
            '[unclosed bracket',
            '[unclosed bracket'  # Leave unclosed single brackets as is
        ),
        (
            'existing [[double]] with [single]',
            'existing [[double]] with [[single]]'
        )
    ]
    
    print("\n🧪 Testing bracket link fixes:")
    print("----------------------------")
    for i, (input_str, expected) in enumerate(test_cases, 1):
        result = fix_bracket_links(input_str)
        success = result == expected
        print(f"\nTest {i}:")
        print(f"Input:    {input_str}")
        print(f"Output:   {result}")
        print(f"Expected: {expected}")
        print(f"Result: {'✅ Pass' if success else '❌ Fail'}")
    print("----------------------------\n")

def main(project_path: str, model_name: str):
    try:
        print(f"\n🚀 Starting research request extraction process")
        print(f"   Project path: {project_path}")
        print(f"   Model: {model_name}\n")
        
        # Start Ollama server
        start_ollama_server()
        
        # Ensure model is available
        if not ensure_model_available(model_name):
            print("❌ Failed to ensure model availability. Exiting.")
            return
        
        # Initialize Ollama client
        client = Client()
        
        # Define input and output paths
        input_file = os.path.join(project_path, "outputs", "conversations", "conversations_list.json")
        output_dir = os.path.join(project_path, "outputs", "research_requests")
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Read conversations
        print(f"📖 Reading conversations from: {input_file}")
        with open(input_file, 'r', encoding='utf-8') as f:
            conversations = json.load(f)
        
        # Process each conversation with progress bar
        all_requests = []
        with tqdm(total=len(conversations), desc="Processing conversations") as pbar:
            for conv_name, conv_text in conversations.items():
                print(f"\n\n{'='*80}")
                print(f"Processing conversation: {conv_name}")
                print(f"{'='*80}\n")
                
                source_info = {
                    "conversation_file": conv_name,
                    "chunk_id": "full"
                }
                
                requests = process_conversation(conv_text, client, model_name, source_info)
                all_requests.extend(requests)
                pbar.update(1)
        
        # Add debug info
        print(f"Found {len(conversations)} conversations to process")
        print(f"Generated {len(all_requests)} total requests")
        
        if not all_requests:
            print("\n❌ No valid requests were generated. Check the errors above.")
            return
        
        # Save research requests and create indexes
        print("\n💾 Saving research requests and creating indexes...")
        save_obsidian_files(all_requests, output_dir)
        
        print(f"\n✅ Successfully processed {len(all_requests)} requests")
        print(f"   Files saved in: {output_dir}")
        print(f"   - Individual markdown files")
        print(f"   - _request_index.json")
        print(f"   - _research_requests_index.md")
        
    except Exception as e:
        print(f"\n❌ An error occurred: {str(e)}")

if __name__ == "__main__":
    test_fix_bracket_links()
    parser = argparse.ArgumentParser(description='Process conversations into Obsidian vault research requests.')
    parser.add_argument('--project_path', type=str, default=DEFAULT_PROJECT_PATH,
                      help=f'Path to the project directory (default: {DEFAULT_PROJECT_PATH})')
    parser.add_argument('--model_name', type=str, default=DEFAULT_MODEL_NAME,
                      help=f'Name of the Ollama model to use (default: {DEFAULT_MODEL_NAME})')

    args = parser.parse_args()
    main(args.project_path, args.model_name) 