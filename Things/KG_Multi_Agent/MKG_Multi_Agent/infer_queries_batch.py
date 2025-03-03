# !pip install ollama
# install ollama from https://ollama.ai/download
# !pip install tqdm
# !pip install pyyaml
# !pip install plotly
# !pip install "plotly[express]" networkx matplotlib
# !pip install spacy
# python -m spacy download en_core_web_sm

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
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.colors import LinearSegmentedColormap
import spacy

# Configuration
force_links = True  # force [[links]] to be created during extraction
DEFAULT_PROJECT_PATH = "./Things/KG_Multi_Agent/MKG_Multi_Agent/test1"
DEFAULT_MODEL_NAME = "llama3.2"  # Choose one model name

# Load spaCy model for NER and phrase detection
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    print("Downloading spaCy model...")
    subprocess.run([sys.executable, "-m", "spacy", "download", "en_core_web_sm"])
    nlp = spacy.load("en_core_web_sm")

def enhance_text_with_links(text: str) -> str:
    """
    Process text to add [[links]] around important phrases using spaCy.
    This includes:
    1. Named entities
    2. Noun phrases
    3. Technical terms
    4. Common research-related phrases
    """
    # Skip if text is already heavily linked
    if text.count('[[') > len(text.split()) / 4:  # If more than 25% of words are linked
        return text
        
    doc = nlp(text)
    
    # Collect all spans that should be linked
    spans_to_link = []
    
    # Add named entities
    spans_to_link.extend([(ent.start_char, ent.end_char, ent.text) for ent in doc.ents])
    
    # Add noun chunks (phrases)
    spans_to_link.extend([
        (chunk.start_char, chunk.end_char, chunk.text) 
        for chunk in doc.noun_chunks 
        if len(chunk.text.split()) > 1  # Only multi-word phrases
    ])
    
    # Add custom research-related terms
    research_terms = [
        "research", "study", "analysis", "methodology", "framework",
        "system", "model", "algorithm", "data", "results", "findings",
        "implementation", "development", "design", "architecture",
        "evaluation", "testing", "validation", "performance", "efficiency",
        "optimization", "integration", "interface", "component", "module",
        "process", "workflow", "pipeline", "infrastructure", "platform",
        "knowledge graph", "neural network", "machine learning", "artificial intelligence",
        "database", "query", "api", "service", "protocol", "standard",
        "test subject", "core team", "program", "application"
    ]
    
    for term in research_terms:
        for match in re.finditer(r'\b' + re.escape(term) + r'\b', text, re.IGNORECASE):
            spans_to_link.append((match.start(), match.end(), match.group()))
    
    # Sort spans and merge overlapping ones
    spans_to_link.sort(key=lambda x: x[0])
    merged_spans = []
    for span in spans_to_link:
        if not merged_spans or span[0] > merged_spans[-1][1]:
            merged_spans.append(span)
        else:
            # Merge overlapping spans
            last_span = merged_spans[-1]
            merged_spans[-1] = (
                last_span[0],
                max(last_span[1], span[1]),
                text[last_span[0]:max(last_span[1], span[1])]
            )
    
    # Apply links from end to start to preserve character positions
    result = text
    for start, end, phrase in reversed(merged_spans):
        # Skip if already within brackets
        if (start > 1 and result[start-2:start] == '[[') or \
           (end < len(result)-1 and result[end:end+2] == ']]'):
            continue
        
        # Skip common words and short phrases
        if len(phrase.split()) == 1 and phrase.lower() in {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to',
            'for', 'of', 'with', 'by', 'from', 'up', 'down', 'over', 'under'
        }:
            continue
            
        # Add brackets
        result = result[:start] + '[[' + phrase + ']]' + result[end:]
    
    return result

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
    
    # Enhance text fields with links if force_links is enabled
    hypothesis = request["hypothesis"]
    rationale = request["rationale"]
    impact = request["impact"]
    
    if force_links:
        hypothesis = enhance_text_with_links(hypothesis)
        rationale = enhance_text_with_links(rationale)
        impact = enhance_text_with_links(impact)
    
    # Create comprehensive frontmatter with all fields
    frontmatter = {
        # Source information
        "source_conversation": source_info["conversation_file"],
        "source_chunk": source_info.get("chunk_id", "full"),
        
        # Core fields from request
        "type": request["intent"].strip("[]"),
        "hypothesis": hypothesis,
        "rationale": rationale,
        "impact": impact,
        
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

# Research Request: {hypothesis}

## Created
{timestamp.strftime("%Y-%m-%d %H:%M:%S")}

## Hypothesis
{hypothesis}

## Context and Rationale
{rationale}

## Expected Impact
{impact}

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
    """Create and save various statistical visualizations of the research requests data using matplotlib and plotly."""
    # Create visualizations and statistics directories
    viz_dir = os.path.join(output_dir, "visualizations")
    stats_dir = os.path.join(output_dir, "statistics")
    os.makedirs(viz_dir, exist_ok=True)
    os.makedirs(stats_dir, exist_ok=True)

    # Calculate basic statistics
    agent_counts = Counter()
    tag_counts = Counter()
    total_requests = len(requests_data)
    
    # Create data structures for Sankey diagram
    agent_tag_flows = defaultdict(lambda: defaultdict(int))
    
    for request in requests_data:
        agents = request['metadata'].get('agents', [])
        tags = request['metadata'].get('tags', [])
        
        # Update counts
        for agent in agents:
            agent_counts[agent] = agent_counts.get(agent, 0) + 1
            # Update agent-tag flows
            for tag in tags:
                agent_tag_flows[agent][tag] += 1
        
        for tag in tags:
            tag_counts[tag] = tag_counts.get(tag, 0) + 1

    # Create Sankey diagram
    agents = list(agent_counts.keys())
    tags = list(tag_counts.keys())
    
    # Create source, target, and value lists for Sankey
    source = []
    target = []
    value = []
    
    # Create node labels
    node_labels = agents + tags
    
    # Create mapping of names to indices
    name_to_index = {name: i for i, name in enumerate(node_labels)}
    
    # Add flows from agents to tags
    for agent in agents:
        for tag in tags:
            if agent_tag_flows[agent][tag] > 0:
                source.append(name_to_index[agent])
                target.append(name_to_index[tag])
                value.append(agent_tag_flows[agent][tag])
    
    # Create Sankey diagram
    fig = go.Figure(data=[go.Sankey(
        node=dict(
            pad=15,
            thickness=20,
            line=dict(color="black", width=0.5),
            label=node_labels,
            color=["#ADD8E6"] * len(agents) + ["#90EE90"] * len(tags)  # Light blue for agents, light green for tags
        ),
        link=dict(
            source=source,
            target=target,
            value=value,
            color=["rgba(0,0,255,0.2)"] * len(source)  # Semi-transparent blue
        )
    )])
    
    fig.update_layout(
        title_text="Agent-Tag Flow Diagram",
        font_size=10,
        height=800
    )
    
    fig.write_html(os.path.join(viz_dir, "agent_tag_flow.html"))

    # Continue with existing matplotlib visualizations...
    # 1. Agent Participation Bar Chart
    plt.figure(figsize=(12, 6))
    participation_rates = [count/total_requests * 100 for count in agent_counts.values()]
    
    bars = plt.bar(agents, participation_rates)
    plt.title('Agent Participation in Research Requests', pad=20)
    plt.xlabel('Agent')
    plt.ylabel('Participation Rate (%)')
    plt.xticks(rotation=45, ha='right')
    
    # Add value annotations
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}%',
                ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(os.path.join(viz_dir, "agent_participation.png"), dpi=300, bbox_inches='tight')
    plt.close()

    # 2. Tag Usage Bar Chart (Top 15 tags)
    plt.figure(figsize=(12, 6))
    top_tags = dict(sorted(tag_counts.items(), key=lambda x: x[1], reverse=True)[:15])
    
    bars = plt.bar(top_tags.keys(), [count/total_requests * 100 for count in top_tags.values()])
    plt.title('Top 15 Tags Usage Distribution', pad=20)
    plt.xlabel('Tag')
    plt.ylabel('Usage Rate (%)')
    plt.xticks(rotation=45, ha='right')
    
    # Add value annotations
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}%',
                ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(os.path.join(viz_dir, "tag_usage.png"), dpi=300, bbox_inches='tight')
    plt.close()

    # 3. Time Series of Request Activity
    dates = [request['timestamp'] for request in requests_data]
    date_counts = Counter(date.date() for date in dates)
    
    plt.figure(figsize=(12, 6))
    plt.plot(sorted(date_counts.keys()), 
            [date_counts[date] for date in sorted(date_counts.keys())],
            marker='o')
    
    plt.title('Research Request Activity Over Time', pad=20)
    plt.xlabel('Date')
    plt.ylabel('Number of Requests')
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    plt.gcf().autofmt_xdate()  # Rotation and alignment of tick labels
    
    # Add value annotations
    for x, y in zip(sorted(date_counts.keys()), [date_counts[date] for date in sorted(date_counts.keys())]):
        plt.annotate(str(y), (x, y), textcoords="offset points", xytext=(0,10), ha='center')
    
    plt.tight_layout()
    plt.savefig(os.path.join(viz_dir, "request_timeline.png"), dpi=300, bbox_inches='tight')
    plt.close()

    # 4. Agent-Tag Heatmap
    agent_tag_matrix = defaultdict(lambda: defaultdict(int))
    for request in requests_data:
        for agent in request['metadata'].get('agents', []):
            for tag in request['metadata'].get('tags', []):
                agent_tag_matrix[agent][tag] += 1

    # Convert to matrix format
    agents = sorted(agent_counts.keys())
    tags = sorted(tag_counts.keys())
    heatmap_data = [[agent_tag_matrix[agent][tag] for tag in tags] for agent in agents]

    plt.figure(figsize=(15, 8))
    plt.imshow(heatmap_data, aspect='auto', cmap='viridis')
    plt.colorbar(label='Number of Collaborations')
    
    plt.title('Agent-Tag Collaboration Heatmap', pad=20)
    plt.xlabel('Tags')
    plt.ylabel('Agents')
    
    # Customize tick labels
    plt.xticks(range(len(tags)), tags, rotation=45, ha='right')
    plt.yticks(range(len(agents)), agents)
    
    # Add value annotations where count > 0
    for i in range(len(agents)):
        for j in range(len(tags)):
            if heatmap_data[i][j] > 0:
                plt.text(j, i, str(heatmap_data[i][j]),
                        ha='center', va='center',
                        color='white' if heatmap_data[i][j] > 2 else 'black')
    
    plt.tight_layout()
    plt.savefig(os.path.join(viz_dir, "agent_tag_heatmap.png"), dpi=300, bbox_inches='tight')
    plt.close()

    # Save detailed statistics to files
    stats = {
        'summary': {
            'total_requests': total_requests,
            'unique_agents': len(agent_counts),
            'unique_tags': len(tag_counts),
            'avg_agents_per_request': sum(len(r['metadata'].get('agents', [])) for r in requests_data) / total_requests,
            'avg_tags_per_request': sum(len(r['metadata'].get('tags', [])) for r in requests_data) / total_requests
        },
        'agent_participation': {
            agent: {
                'count': count,
                'percentage': (count/total_requests) * 100
            } for agent, count in agent_counts.items()
        },
        'tag_usage': {
            tag: {
                'count': count,
                'percentage': (count/total_requests) * 100
            } for tag, count in tag_counts.items()
        },
        'temporal_stats': {
            str(date): count for date, count in date_counts.items()
        }
    }

    # Save statistics as JSON
    with open(os.path.join(stats_dir, 'statistics.json'), 'w', encoding='utf-8') as f:
        json.dump(stats, f, indent=2, default=str)

    # Save statistics as markdown for better readability
    with open(os.path.join(stats_dir, 'statistics.md'), 'w', encoding='utf-8') as f:
        f.write("# Research Request Statistics\n\n")
        
        f.write("## Summary Statistics\n")
        f.write(f"- Total Requests: {stats['summary']['total_requests']}\n")
        f.write(f"- Unique Agents: {stats['summary']['unique_agents']}\n")
        f.write(f"- Unique Tags: {stats['summary']['unique_tags']}\n")
        f.write(f"- Average Agents per Request: {stats['summary']['avg_agents_per_request']:.2f}\n")
        f.write(f"- Average Tags per Request: {stats['summary']['avg_tags_per_request']:.2f}\n\n")
        
        f.write("## Agent Participation\n")
        for agent, data in sorted(stats['agent_participation'].items(), key=lambda x: x[1]['count'], reverse=True):
            f.write(f"- {agent}: {data['count']} requests ({data['percentage']:.1f}%)\n")
        
        f.write("\n## Tag Usage\n")
        for tag, data in sorted(stats['tag_usage'].items(), key=lambda x: x[1]['count'], reverse=True):
            f.write(f"- {tag}: {data['count']} requests ({data['percentage']:.1f}%)\n")
        
        f.write("\n## Daily Activity\n")
        for date, count in sorted(stats['temporal_stats'].items()):
            f.write(f"- {date}: {count} requests\n")

def create_network_visualizations(requests_data: List[Dict], output_dir: str):
    """Create and save network visualizations of the research requests data using both matplotlib and plotly."""
    viz_dir = os.path.join(output_dir, "visualizations")
    stats_dir = os.path.join(output_dir, "statistics")
    os.makedirs(viz_dir, exist_ok=True)
    os.makedirs(stats_dir, exist_ok=True)
    
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
    node_degrees = dict(G.degree())
    
    # Calculate bridging coefficients
    def calculate_bridging_coefficient(G, node):
        neighbors = set(G.neighbors(node))
        if len(neighbors) <= 1:
            return 0.0
        
        # Count edges between neighbors
        neighbor_edges = sum(1 for n1 in neighbors for n2 in neighbors if G.has_edge(n1, n2))
        max_possible_edges = len(neighbors) * (len(neighbors) - 1) / 2
        
        if max_possible_edges == 0:
            return 0.0
        
        # Bridging coefficient is 1 minus the ratio of actual to possible edges between neighbors
        return 1.0 - (neighbor_edges / max_possible_edges)
    
    bridging_coefficients = {node: calculate_bridging_coefficient(G, node) for node in G.nodes()}
    
    # Create matplotlib visualization
    plt.figure(figsize=(15, 10))
    
    # Use spring layout with adjusted parameters for better spacing
    pos = nx.spring_layout(G, k=1.5, iterations=50)
    
    # Draw edges with different colors for different types
    agent_request_edges = [(u, v) for (u, v, d) in G.edges(data=True) if d.get('edge_type') == 'agent_request']
    tag_request_edges = [(u, v) for (u, v, d) in G.edges(data=True) if d.get('edge_type') == 'tag_request']
    
    nx.draw_networkx_edges(G, pos, edgelist=agent_request_edges, alpha=0.2, edge_color='blue')
    nx.draw_networkx_edges(G, pos, edgelist=tag_request_edges, alpha=0.2, edge_color='orange')
    
    # Draw nodes with different colors and sizes based on type and centrality
    agent_nodes = [n for n, attr in G.nodes(data=True) if attr.get('node_type') == 'agent']
    tag_nodes = [n for n, attr in G.nodes(data=True) if attr.get('node_type') == 'tag']
    request_nodes = [n for n, attr in G.nodes(data=True) if attr.get('node_type') == 'request']
    
    # Draw nodes
    nx.draw_networkx_nodes(G, pos, nodelist=agent_nodes, node_color='lightblue',
                          node_size=[3000 * degree_centrality[n] for n in agent_nodes],
                          label='Agents')
    nx.draw_networkx_nodes(G, pos, nodelist=tag_nodes, node_color='lightgreen',
                          node_size=[3000 * degree_centrality[n] for n in tag_nodes],
                          label='Tags')
    nx.draw_networkx_nodes(G, pos, nodelist=request_nodes, node_color='lightgray',
                          node_size=[2000 * degree_centrality[n] for n in request_nodes],
                          label='Requests')
    
    # Add labels with smaller font size for better readability
    nx.draw_networkx_labels(G, pos, font_size=8)
    
    plt.title('Research Request Network\nNode size represents degree centrality', pad=20)
    plt.legend()
    plt.axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(viz_dir, "network_visualization.png"), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Create interactive Plotly visualization
    edge_x = []
    edge_y = []
    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])

    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=0.5, color='#888'),
        hoverinfo='none',
        mode='lines')

    node_x = []
    node_y = []
    node_text = []
    node_colors = []
    node_sizes = []

    for node in G.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        
        node_type = G.nodes[node].get('node_type', 'unknown')
        node_info = (
            f"Node: {node}<br>"
            f"Type: {node_type}<br>"
            f"Degree: {node_degrees[node]}<br>"
            f"Degree Centrality: {degree_centrality[node]:.3f}<br>"
            f"Betweenness Centrality: {betweenness_centrality[node]:.3f}<br>"
            f"Bridging Coefficient: {bridging_coefficients[node]:.3f}"
        )
        
        if node_type == 'request':
            node_info += f"<br>Hypothesis: {G.nodes[node].get('hypothesis', 'N/A')}"
        
        node_text.append(node_info)
        
        # Set node colors based on type
        if node_type == 'agent':
            node_colors.append('#ADD8E6')  # lightblue
            node_sizes.append(20)
        elif node_type == 'tag':
            node_colors.append('#90EE90')  # lightgreen
            node_sizes.append(15)
        else:  # request
            node_colors.append('#D3D3D3')  # lightgray
            node_sizes.append(10)

    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers+text',
        hoverinfo='text',
        text=[node if G.nodes[node].get('node_type') in ['agent', 'tag'] else '' for node in G.nodes()],  # Only show labels for agents and tags
        textposition="top center",
        hovertext=node_text,
        marker=dict(
            color=node_colors,
            size=node_sizes,
            line_width=2))

    fig = go.Figure(data=[edge_trace, node_trace],
                   layout=go.Layout(
                       title={'text': 'Interactive Research Request Network',
                             'font': {'size': 16}},
                       showlegend=False,
                       hovermode='closest',
                       margin=dict(b=20,l=5,r=5,t=40),
                       xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                       yaxis=dict(showgrid=False, zeroline=False, showticklabels=False))
                   )

    fig.write_html(os.path.join(viz_dir, "network_visualization.html"))
    
    # Save network statistics
    stats = {
        'network_summary': {
            'number_of_nodes': len(G.nodes()),
            'number_of_edges': len(G.edges()),
            'average_degree': sum(dict(G.degree()).values()) / len(G.nodes()),
            'network_density': nx.density(G),
            'average_clustering_coefficient': nx.average_clustering(G),
        },
        'centrality_metrics': {
            'degree_centrality': {node: round(cent, 3) for node, cent in degree_centrality.items()},
            'betweenness_centrality': {node: round(cent, 3) for node, cent in betweenness_centrality.items()},
            'node_degrees': node_degrees,
            'bridging_coefficients': {node: round(coef, 3) for node, coef in bridging_coefficients.items()}
        },
        'node_statistics': {
            'agents': len(agent_nodes),
            'tags': len(tag_nodes),
            'requests': len(request_nodes)
        },
        'edge_statistics': {
            'agent_request_connections': len(agent_request_edges),
            'tag_request_connections': len(tag_request_edges)
        }
    }
    
    # Save network statistics as JSON
    with open(os.path.join(stats_dir, 'network_statistics.json'), 'w', encoding='utf-8') as f:
        json.dump(stats, f, indent=2, default=str)
    
    # Save network statistics as markdown
    with open(os.path.join(stats_dir, 'network_statistics.md'), 'w', encoding='utf-8') as f:
        f.write("# Network Analysis Statistics\n\n")
        
        f.write("## Network Summary\n")
        f.write(f"- Number of Nodes: {stats['network_summary']['number_of_nodes']}\n")
        f.write(f"- Number of Edges: {stats['network_summary']['number_of_edges']}\n")
        f.write(f"- Average Degree: {stats['network_summary']['average_degree']:.2f}\n")
        f.write(f"- Network Density: {stats['network_summary']['network_density']:.3f}\n")
        f.write(f"- Average Clustering Coefficient: {stats['network_summary']['average_clustering_coefficient']:.3f}\n\n")
        
        f.write("## Node Distribution\n")
        f.write(f"- Agents: {stats['node_statistics']['agents']}\n")
        f.write(f"- Tags: {stats['node_statistics']['tags']}\n")
        f.write(f"- Requests: {stats['node_statistics']['requests']}\n\n")
        
        f.write("## Edge Distribution\n")
        f.write(f"- Agent-Request Connections: {stats['edge_statistics']['agent_request_connections']}\n")
        f.write(f"- Tag-Request Connections: {stats['edge_statistics']['tag_request_connections']}\n\n")
        
        f.write("## Node Metrics\n\n")
        
        f.write("### Top Nodes by Degree\n")
        sorted_degrees = sorted(stats['centrality_metrics']['node_degrees'].items(),
                              key=lambda x: x[1], reverse=True)[:10]
        for node, degree in sorted_degrees:
            f.write(f"- {node}: {degree}\n")
        
        f.write("\n### Top Nodes by Degree Centrality\n")
        sorted_degree = sorted(stats['centrality_metrics']['degree_centrality'].items(),
                             key=lambda x: x[1], reverse=True)[:10]
        for node, centrality in sorted_degree:
            f.write(f"- {node}: {centrality:.3f}\n")
        
        f.write("\n### Top Nodes by Betweenness Centrality\n")
        sorted_betweenness = sorted(stats['centrality_metrics']['betweenness_centrality'].items(),
                                  key=lambda x: x[1], reverse=True)[:10]
        for node, centrality in sorted_betweenness:
            f.write(f"- {node}: {centrality:.3f}\n")
        
        f.write("\n### Top Nodes by Bridging Coefficient\n")
        sorted_bridging = sorted(stats['centrality_metrics']['bridging_coefficients'].items(),
                               key=lambda x: x[1], reverse=True)[:10]
        for node, coefficient in sorted_bridging:
            f.write(f"- {node}: {coefficient:.3f}\n")

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

def main(project_path: str, model_name: str, force_links: bool):
    try:
        print(f"\n🚀 Starting research request extraction process")
        print(f"   Project path: {project_path}")
        print(f"   Model: {model_name}")
        print(f"   Force links: {'enabled' if force_links else 'disabled'}\n")
        
        # Update global force_links setting
        globals()['force_links'] = force_links
        
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
    parser.add_argument('--force_links', action='store_true', default=force_links,
                      help='Force creation of [[links]] in hypothesis, rationale, and impact sections')

    args = parser.parse_args()
    main(args.project_path, args.model_name, args.force_links) 