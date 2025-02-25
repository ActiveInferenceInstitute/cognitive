"""
Visualization utilities for knowledge networks.
"""

import re
import networkx as nx
import plotly.graph_objects as go
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple
import yaml

class NetworkVisualizer:
    """Visualizes knowledge networks using networkx and plotly."""
    
    def __init__(self, config_path: str = "config.yaml"):
        """Initialize the network visualizer.
        
        Args:
            config_path: Path to configuration file
        """
        self.config = self._load_config(config_path)
        self.knowledge_base_path = Path(self.config['paths']['knowledge_base'])
        self.link_pattern = re.compile(r'\[\[(.*?)\]\]')
        
    def _load_config(self, config_path: str) -> Dict:
        """Load configuration from YAML file."""
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    
    def _extract_links(self, content: str) -> Set[str]:
        """Extract links from markdown content."""
        return set(self.link_pattern.findall(content))
    
    def _get_node_type(self, file_path: Path) -> str:
        """Determine node type from file path."""
        return file_path.parent.name.rstrip('s')
    
    def _get_node_metadata(self, file_path: Path) -> Dict:
        """Extract node metadata from file."""
        with open(file_path, 'r') as f:
            content = f.read()
            
        # Extract YAML frontmatter
        if content.startswith('---'):
            _, frontmatter, _ = content.split('---', 2)
            metadata = yaml.safe_load(frontmatter)
        else:
            metadata = {}
            
        metadata['type'] = self._get_node_type(file_path)
        metadata['links'] = self._extract_links(content)
        return metadata
    
    def build_network(self) -> nx.Graph:
        """Build network from knowledge base."""
        G = nx.Graph()
        
        # Add nodes
        for file_path in self.knowledge_base_path.rglob('*.md'):
            node_id = file_path.stem
            metadata = self._get_node_metadata(file_path)
            G.add_node(node_id, **metadata)
            
            # Add edges from links
            for link in metadata['links']:
                G.add_edge(node_id, link)
        
        return G
    
    def _get_node_positions(self, G: nx.Graph) -> Dict:
        """Get node positions using force-directed layout."""
        return nx.spring_layout(G)
    
    def _get_node_colors(self, G: nx.Graph) -> List[str]:
        """Get node colors based on type."""
        color_map = {
            'agent': '#FF6B6B',
            'belief': '#4ECDC4',
            'goal': '#45B7D1',
            'action': '#96CEB4',
            'observation': '#FFEEAD',
            'relationship': '#D4A5A5'
        }
        return [color_map.get(G.nodes[node]['type'], '#CCCCCC') for node in G.nodes()]
    
    def visualize(self, 
                 output_path: Optional[str] = None,
                 width: int = 1200,
                 height: int = 800) -> go.Figure:
        """Create interactive network visualization.
        
        Args:
            output_path: Path to save HTML output (optional)
            width: Plot width in pixels
            height: Plot height in pixels
            
        Returns:
            Plotly figure object
        """
        # Build network
        G = self.build_network()
        pos = self._get_node_positions(G)
        node_colors = self._get_node_colors(G)
        
        # Create edge trace
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
        
        # Create node trace
        node_x = [pos[node][0] for node in G.nodes()]
        node_y = [pos[node][1] for node in G.nodes()]
        
        node_trace = go.Scatter(
            x=node_x, y=node_y,
            mode='markers+text',
            hoverinfo='text',
            marker=dict(
                size=20,
                color=node_colors,
                line_width=2,
                line=dict(color='white')
            ),
            text=list(G.nodes()),
            textposition="top center"
        )
        
        # Create figure
        fig = go.Figure(
            data=[edge_trace, node_trace],
            layout=go.Layout(
                title='Knowledge Network',
                showlegend=False,
                hovermode='closest',
                margin=dict(b=20,l=5,r=5,t=40),
                annotations=[],
                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                width=width,
                height=height
            )
        )
        
        # Save if output path provided
        if output_path:
            fig.write_html(output_path)
        
        return fig

def main():
    """CLI entry point."""
    import click
    
    @click.command()
    @click.option('--config', '-c', default='config.yaml', help='Path to config file')
    @click.option('--output', '-o', help='Path to save visualization HTML')
    @click.option('--width', '-w', default=1200, help='Plot width in pixels')
    @click.option('--height', '-h', default=800, help='Plot height in pixels')
    def visualize(config: str, output: str, width: int, height: int):
        """Create knowledge network visualization."""
        try:
            viz = NetworkVisualizer(config)
            fig = viz.visualize(output, width, height)
            if not output:
                fig.show()
        except Exception as e:
            click.echo(f"Error creating visualization: {str(e)}", err=True)
            return 1
    
    visualize()

if __name__ == '__main__':
    main() 