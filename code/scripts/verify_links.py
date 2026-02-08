import os
import re

def verify_links(root_dir):
    link_pattern = re.compile(r'\[\[(.*?)\]\]')
    broken_links = []
    
    # Store all markdown files for fuzzy matching
    all_files = {}
    for dirpath, _, filenames in os.walk(root_dir):
        for f in filenames:
            if f.endswith('.md'):
                rel_path = os.path.relpath(os.path.join(dirpath, f), root_dir)
                name_no_ext = os.path.splitext(f)[0]
                if name_no_ext not in all_files:
                    all_files[name_no_ext] = []
                all_files[name_no_ext].append(rel_path)

    print(f"Index built with {len(all_files)} unique filenames.")

    for dirpath, _, filenames in os.walk(root_dir):
        for filename in filenames:
            if not filename.endswith('.md'):
                continue
            
            filepath = os.path.join(dirpath, filename)
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    content = f.read()
            except Exception as e:
                print(f"Error reading {filepath}: {e}")
                continue

            # Remove code blocks and inline code to avoid false positives
            # Remove multi-line code blocks
            content_no_code = re.sub(r'```[\s\S]*?```', '', content)
            # Remove inline code
            content_no_code = re.sub(r'`[^`\n]+`', '', content_no_code)

            matches = link_pattern.findall(content_no_code)
            for match in matches:
                # Ignore matrix notation like [[1, 2], [3, 4]]
                if re.match(r'^[\d,\s\[\]\.]+$', match):
                    continue

                # Handle aliases [[link|alias]]
                target = match.split('|')[0].strip()
                
                # Handle anchor links [[link#anchor]]
                target_file = target.split('#')[0].strip()
                
                if not target_file:
                    continue

                # 1. Check if file exists relative to current file
                current_dir = os.path.dirname(filepath)
                
                possible_paths = []
                
                # Handle directory links
                if target_file.endswith('/'):
                    possible_paths.append(os.path.join(current_dir, target_file, 'README.md'))
                    possible_paths.append(os.path.join(current_dir, target_file, 'index.md'))
                    possible_paths.append(os.path.join(root_dir, target_file, 'README.md'))
                    possible_paths.append(os.path.join(root_dir, target_file, 'index.md'))
                else:
                    possible_paths.append(os.path.join(current_dir, target_file))
                    possible_paths.append(os.path.join(current_dir, target_file + '.md'))
                    possible_paths.append(os.path.join(root_dir, target_file))
                    possible_paths.append(os.path.join(root_dir, target_file + '.md'))
                    # Also try resolving ../
                    possible_paths.append(os.path.normpath(os.path.join(current_dir, target_file)))
                    possible_paths.append(os.path.normpath(os.path.join(current_dir, target_file + '.md')))

                found = False
                for p in possible_paths:
                    if os.path.exists(p) and os.path.isfile(p):
                        found = True
                        break
                
                # 2. Check Obsidian fuzzy match (filename only)
                if not found:
                    basename = os.path.basename(target_file)
                    name_no_ext = os.path.splitext(basename)[0]
                    if name_no_ext in all_files:
                        found = True

                if not found:
                    broken_links.append({
                        'source': os.path.relpath(filepath, root_dir),
                        'link': match,
                        'target': target_file
                    })

    return broken_links

if __name__ == "__main__":
    import sys
    root = sys.argv[1] if len(sys.argv) > 1 else "."
    broken = verify_links(root)
    
    if broken:
        print(f"Found {len(broken)} broken links:")
        for b in broken:
            print(f"  Source: {b['source']}")
            print(f"  Link: [[{b['link']}]]")
            print(f"  Target: {b['target']}")
            print("-" * 20)
        sys.exit(1)
    else:
        print("All links verified successfully!")
