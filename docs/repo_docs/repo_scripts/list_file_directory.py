import os
import json
import csv
import mimetypes
import re
from pathlib import Path
from typing import Dict, List, Set
from datetime import datetime
from tqdm import tqdm
import time

def extract_wikilinks(content: str) -> Set[str]:
    """Extract all [[wikilinks]] from content.
    
    Args:
        content (str): File content to analyze
        
    Returns:
        Set[str]: Set of unique wikilink targets
    """
    # Match [[link]] or [[link|alias]] patterns
    pattern = r'\[\[(.*?)(?:\|.*?)?\]\]'
    matches = re.findall(pattern, content)
    return set(matches)

def get_file_info(file_path: Path) -> Dict:
    """Get metadata and content info for a file.
    
    Args:
        file_path (Path): Path to the file
        
    Returns:
        Dict: File metadata including size, type, and wikilinks
    """
    info = {
        'name': file_path.name,
        'extension': file_path.suffix,
        'size': file_path.stat().st_size,
        'mime_type': mimetypes.guess_type(file_path)[0],
        'relative_path': str(file_path),
        'modified_time': datetime.fromtimestamp(file_path.stat().st_mtime).isoformat(),
        'wikilinks': set()
    }
    
    # Extract wikilinks from text files
    if file_path.suffix in ['.md', '.txt']:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                info['wikilinks'] = extract_wikilinks(content)
        except Exception as e:
            print(f"Error reading {file_path}: {e}")
            
    return info

def list_files_in_directory(directory_path: str) -> Dict:
    """Lists all files in the given directory and prints their names, sizes, types, and complete file structure + paths.
    
    This function recursively traverses a directory and collects information about each file, including:
    - File name and extension
    - File size in bytes
    - File type/MIME type
    - Full path relative to the root directory
    - Any Obsidian-style [[wikilinks]] found in the file content
    
    Args:
        directory_path (str): Path to the directory to analyze
        
    Returns:
        dict: Dictionary containing the file structure and metadata
    """
    root_path = Path(directory_path)
    
    print("\n📂 Scanning directory structure...")
    
    # Initialize structure
    structure = {
        'root_path': str(root_path),
        'files': {},
        'directories': {},
        'wikilink_graph': {},
        'stats': {
            'total_files': 0,
            'total_size': 0,
            'file_types': {},
            'wikilinks': set()
        }
    }
    
    # Get total file count for progress bar
    total_files = sum(1 for _ in root_path.rglob('*') if _.is_file())
    
    # Recursively process directory
    with tqdm(total=total_files, desc="Processing files", unit="file") as pbar:
        for path in root_path.rglob('*'):
            if path.is_file():
                # Get relative path from root
                rel_path = path.relative_to(root_path)
                parent_dir = str(rel_path.parent)
                
                # Get file info
                file_info = get_file_info(path)
                
                # Update structure
                if parent_dir not in structure['directories']:
                    structure['directories'][parent_dir] = []
                structure['directories'][parent_dir].append(file_info['name'])
                
                # Store file info
                structure['files'][str(rel_path)] = file_info
                
                # Update stats
                structure['stats']['total_files'] += 1
                structure['stats']['total_size'] += file_info['size']
                ext = file_info['extension']
                structure['stats']['file_types'][ext] = structure['stats']['file_types'].get(ext, 0) + 1
                structure['stats']['wikilinks'].update(file_info['wikilinks'])
                
                # Update wikilink graph
                if file_info['wikilinks']:
                    structure['wikilink_graph'][str(rel_path)] = list(file_info['wikilinks'])
                
                pbar.update(1)
    
    return structure

def export_json(structure: Dict, output_dir: Path) -> None:
    """Export the directory structure as JSON.
    
    Args:
        structure (Dict): Directory structure
        output_dir (Path): Output directory path
    """
    # Convert sets to lists for JSON serialization
    json_structure = {
        'root_path': structure['root_path'],
        'files': {
            k: {
                **v, 
                'wikilinks': list(v['wikilinks'])
            } for k, v in structure['files'].items()
        },
        'directories': structure['directories'],
        'wikilink_graph': structure['wikilink_graph'],
        'stats': {
            **structure['stats'],
            'wikilinks': list(structure['stats']['wikilinks'])
        }
    }
    
    output_file = output_dir / 'file_structure.json'
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(json_structure, f, indent=2)

def export_csv(structure: Dict, output_dir: Path) -> None:
    """Export file information as CSV.
    
    Args:
        structure (Dict): Directory structure
        output_dir (Path): Output directory path
    """
    # Export files data
    files_output = output_dir / 'files.csv'
    with open(files_output, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['path', 'name', 'extension', 'size', 'mime_type', 'modified_time', 'wikilinks'])
        for path, info in structure['files'].items():
            writer.writerow([
                path,
                info['name'],
                info['extension'],
                info['size'],
                info['mime_type'],
                info['modified_time'],
                '|'.join(info['wikilinks'])
            ])
    
    # Export wikilink graph
    links_output = output_dir / 'wikilinks.csv'
    with open(links_output, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['source', 'target'])
        for source, targets in structure['wikilink_graph'].items():
            for target in targets:
                writer.writerow([source, target])

def export_txt(structure: Dict, output_dir: Path) -> None:
    """Export directory structure as formatted text.
    
    Args:
        structure (Dict): Directory structure
        output_dir (Path): Output directory path
    """
    output_file = output_dir / 'file_structure.txt'
    with open(output_file, 'w', encoding='utf-8') as f:
        # Write header
        f.write("=== Directory Structure Analysis ===\n\n")
        
        # Write basic stats
        f.write(f"Root path: {structure['root_path']}\n")
        f.write(f"Total files: {structure['stats']['total_files']}\n")
        f.write(f"Total size: {structure['stats']['total_size']} bytes\n\n")
        
        # Write file types
        f.write("File types:\n")
        for ext, count in structure['stats']['file_types'].items():
            f.write(f"  {ext}: {count} files\n")
        
        # Write directory structure
        f.write("\nDirectory structure:\n")
        for dir_path, files in structure['directories'].items():
            f.write(f"\n{dir_path}/\n")
            for file in files:
                f.write(f"  - {file}\n")
        
        # Write wikilink relationships
        f.write("\nWikilink relationships:\n")
        for source, targets in structure['wikilink_graph'].items():
            if targets:
                f.write(f"\n{source} links to:\n")
                for target in targets:
                    f.write(f"  - [[{target}]]\n")

def export_structure(structure: Dict, output_dir: Path) -> None:
    """Export directory structure in multiple formats.
    
    Args:
        structure (Dict): Directory structure
        output_dir (Path): Output directory path
    """
    print("\n💾 Exporting directory structure...")
    
    # Create output directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Export in different formats
    with tqdm(total=3, desc="Generating outputs", unit="file") as pbar:
        export_json(structure, output_dir)
        pbar.update(1)
        
        export_csv(structure, output_dir)
        pbar.update(1)
        
        export_txt(structure, output_dir)
        pbar.update(1)
    
    print(f"\n✨ Exported directory structure to {output_dir}:")
    print(f"  📄 JSON: file_structure.json")
    print(f"  📊 CSV: files.csv, wikilinks.csv")
    print(f"  📝 Text: file_structure.txt")

def print_structure(structure: Dict) -> None:
    """Print a human-readable summary of the directory structure.
    
    Args:
        structure (Dict): Directory structure from list_files_in_directory
    """
    print("\n📊 Directory Structure Analysis")
    print("=" * 40 + "\n")
    
    # Print basic stats
    print(f"📂 Root path: {structure['root_path']}")
    print(f"📁 Total files: {structure['stats']['total_files']}")
    print(f"💾 Total size: {structure['stats']['total_size']:,} bytes")
    
    # Print file types
    print("\n📋 File types:")
    for ext, count in structure['stats']['file_types'].items():
        print(f"  {ext or '(no extension)'}: {count} files")
    
    # Print directory structure
    print("\n🌳 Directory structure:")
    for dir_path, files in structure['directories'].items():
        print(f"\n{dir_path}/")
        for file in files:
            print(f"  - {file}")
    
    # Print wikilink relationships
    print("\n🔗 Wikilink relationships:")
    for source, targets in structure['wikilink_graph'].items():
        if targets:
            print(f"\n{source} links to:")
            for target in targets:
                print(f"  - [[{target}]]")
    
    print("\n" + "=" * 40)

def main():
    """Main function to run the directory analysis."""
    start_time = time.time()
    
    # Get the script's directory
    script_dir = Path(__file__).parent
    
    # Analyze the parent directory (knowledge base root)
    kb_dir = script_dir.parent
    
    print("\n🔍 Knowledge Base Directory Analysis")
    print("=" * 40)
    
    print(f"\n📂 Analyzing directory: {kb_dir}")
    structure = list_files_in_directory(kb_dir)
    
    # Create output directory
    output_dir = script_dir / 'output'
    export_structure(structure, output_dir)
    
    # Also print to console
    print_structure(structure)
    
    end_time = time.time()
    duration = end_time - start_time
    print(f"\n✨ Analysis completed in {duration:.2f}s\n")

if __name__ == "__main__":
    main()