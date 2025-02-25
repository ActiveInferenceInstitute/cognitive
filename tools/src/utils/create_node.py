#!/usr/bin/env python3
"""
Utility script for creating new nodes from templates.
"""

import os
import sys
import yaml
import click
from datetime import datetime
from jinja2 import Environment, FileSystemLoader
from pathlib import Path
from typing import Dict, Optional

class NodeCreator:
    """Creates new nodes from templates."""
    
    def __init__(self, config_path: str = "config.yaml"):
        """Initialize the node creator.
        
        Args:
            config_path: Path to configuration file
        """
        self.config = self._load_config(config_path)
        self.template_path = Path(self.config['paths']['templates'])
        self.knowledge_base_path = Path(self.config['paths']['knowledge_base'])
        
        # Set up Jinja environment
        self.env = Environment(
            loader=FileSystemLoader(self.template_path),
            trim_blocks=True,
            lstrip_blocks=True
        )
    
    def _load_config(self, config_path: str) -> Dict:
        """Load configuration from YAML file."""
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    
    def _generate_id(self, prefix: str) -> str:
        """Generate a unique ID for the node."""
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        return f"{prefix}_{timestamp}"
    
    def _ensure_directory(self, path: Path):
        """Ensure directory exists."""
        path.mkdir(parents=True, exist_ok=True)
    
    def create_node(self, 
                   node_type: str, 
                   name: str, 
                   template_vars: Optional[Dict] = None) -> Path:
        """Create a new node from template.
        
        Args:
            node_type: Type of node to create
            name: Name of the node
            template_vars: Additional template variables
            
        Returns:
            Path to created node file
        """
        # Load template
        template_name = f"{node_type}_template.md"
        template = self.env.get_template(f"node_templates/{template_name}")
        
        # Prepare variables
        vars_dict = template_vars or {}
        vars_dict.update({
            'node_id': self._generate_id(node_type),
            'node_name': name,
            'date': datetime.now().isoformat(),
            'type': node_type
        })
        
        # Generate content
        content = template.render(**vars_dict)
        
        # Create output file
        output_dir = self.knowledge_base_path / node_type + 's'
        self._ensure_directory(output_dir)
        
        output_file = output_dir / f"{name}.md"
        with open(output_file, 'w') as f:
            f.write(content)
        
        return output_file

@click.command()
@click.option('--type', '-t', required=True, help='Type of node to create')
@click.option('--name', '-n', required=True, help='Name of the node')
@click.option('--config', '-c', default='config.yaml', help='Path to config file')
@click.option('--vars', '-v', multiple=True, help='Additional template variables in key=value format')
def main(type: str, name: str, config: str, vars: tuple):
    """Create a new node from template."""
    # Parse template variables
    template_vars = {}
    for var in vars:
        key, value = var.split('=', 1)
        template_vars[key.strip()] = value.strip()
    
    try:
        creator = NodeCreator(config)
        output_file = creator.create_node(type, name, template_vars)
        click.echo(f"Created node at: {output_file}")
    except Exception as e:
        click.echo(f"Error creating node: {str(e)}", err=True)
        sys.exit(1)

if __name__ == '__main__':
    main() 