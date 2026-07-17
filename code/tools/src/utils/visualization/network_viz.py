"""Deterministic, alias-aware knowledge-network visualization."""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Any

import networkx as nx
import plotly.graph_objects as go
import yaml


class NetworkVisualizer:
    """Build and render a graph from Markdown files in a configured vault."""

    _LINK_PATTERN = re.compile(r"\[\[([^\]]+)\]\]")

    def __init__(self, config_path: str | Path = "config.yaml") -> None:
        self.config_path = Path(config_path).expanduser().resolve()
        self.config = self._load_config(self.config_path)
        paths = self.config.get("paths")
        if not isinstance(paths, dict) or not isinstance(paths.get("knowledge_base"), str):
            raise ValueError("Configuration must define paths.knowledge_base")
        kb_path = Path(paths["knowledge_base"]).expanduser()
        if not kb_path.is_absolute():
            kb_path = self.config_path.parent / kb_path
        self.knowledge_base_path = kb_path.resolve()
        visualization = self.config.get("visualization", {})
        if not isinstance(visualization, dict):
            raise ValueError("visualization must be a mapping")
        self.seed = int(visualization.get("seed", 0))
        self.link_pattern = self._LINK_PATTERN

    @staticmethod
    def _load_config(config_path: Path) -> dict[str, Any]:
        with config_path.open(encoding="utf-8") as config_file:
            config = yaml.safe_load(config_file) or {}
        if not isinstance(config, dict):
            raise ValueError("Configuration must be a YAML mapping")
        return config

    def _extract_links(self, content: str) -> set[str]:
        links: set[str] = set()
        for raw in self.link_pattern.findall(content):
            target = raw.split("|", 1)[0].split("#", 1)[0].strip()
            if target:
                links.add(target)
        return links

    def _get_node_type(self, file_path: Path) -> str:
        directory = file_path.parent.name
        return directory[:-1] if directory.endswith("s") else directory

    def _get_node_metadata(self, file_path: Path) -> dict[str, Any]:
        content = file_path.read_text(encoding="utf-8")
        metadata: dict[str, Any] = {}
        if content.startswith("---"):
            parts = content.split("---", 2)
            if len(parts) == 3:
                parsed = yaml.safe_load(parts[1]) or {}
                if isinstance(parsed, dict):
                    metadata.update(parsed)
        metadata["type"] = self._get_node_type(file_path)
        metadata["links"] = self._extract_links(content)
        return metadata

    def build_network(self) -> nx.Graph:
        """Build a graph whose nodes are canonical relative Markdown paths."""
        graph = nx.Graph()
        if not self.knowledge_base_path.is_dir():
            return graph
        files = sorted(self.knowledge_base_path.rglob("*.md"))
        by_target: dict[str, str] = {}
        metadata_by_node: dict[str, dict[str, Any]] = {}
        for file_path in files:
            node_id = file_path.relative_to(self.knowledge_base_path).with_suffix("").as_posix()
            metadata = self._get_node_metadata(file_path)
            metadata_by_node[node_id] = metadata
            graph.add_node(
                node_id, **{key: value for key, value in metadata.items() if key != "links"}
            )
            for key in {
                node_id,
                file_path.stem,
                file_path.name,
                str(file_path.relative_to(self.knowledge_base_path)),
            }:
                by_target[key.lower()] = node_id
        for node_id, metadata in metadata_by_node.items():
            for target in sorted(metadata["links"]):
                resolved = by_target.get(target.lower())
                if resolved is not None and resolved != node_id:
                    graph.add_edge(node_id, resolved)
        return graph

    def _get_node_positions(self, graph: nx.Graph) -> dict[str, tuple[float, float]]:
        if not graph:
            return {}
        if len(graph) == 1:
            return {next(iter(graph.nodes)): (0.0, 0.0)}
        return nx.spring_layout(graph, seed=self.seed)

    def _get_node_colors(self, graph: nx.Graph) -> list[str]:
        color_map = {
            "agent": "#FF6B6B",
            "belief": "#4ECDC4",
            "goal": "#45B7D1",
            "action": "#96CEB4",
            "observation": "#FFEEAD",
            "relationship": "#D4A5A5",
        }
        return [
            color_map.get(graph.nodes[node].get("type", ""), "#CCCCCC") for node in graph.nodes()
        ]

    def visualize(
        self, output_path: str | Path | None = None, width: int = 1200, height: int = 800
    ) -> go.Figure:
        if width < 1 or height < 1:
            raise ValueError("width and height must be positive")
        graph = self.build_network()
        positions = self._get_node_positions(graph)
        edge_x: list[float | None] = []
        edge_y: list[float | None] = []
        for left, right in sorted(graph.edges()):
            x0, y0 = positions[left]
            x1, y1 = positions[right]
            edge_x.extend([float(x0), float(x1), None])
            edge_y.extend([float(y0), float(y1), None])
        edge_trace = go.Scatter(
            x=edge_x, y=edge_y, line={"width": 0.5, "color": "#888"}, hoverinfo="none", mode="lines"
        )
        node_ids = list(graph.nodes())
        node_trace = go.Scatter(
            x=[positions[node][0] for node in node_ids],
            y=[positions[node][1] for node in node_ids],
            mode="markers+text",
            hoverinfo="text",
            marker={
                "size": 20,
                "color": self._get_node_colors(graph),
                "line": {"width": 2, "color": "white"},
            },
            text=node_ids,
            textposition="top center",
        )
        figure = go.Figure(
            data=[edge_trace, node_trace],
            layout=go.Layout(
                title="Knowledge Network",
                showlegend=False,
                hovermode="closest",
                margin={"b": 20, "l": 5, "r": 5, "t": 40},
                xaxis={"showgrid": False, "zeroline": False, "showticklabels": False},
                yaxis={"showgrid": False, "zeroline": False, "showticklabels": False},
                width=width,
                height=height,
            ),
        )
        if output_path is not None:
            path = Path(output_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            figure.write_html(path)
        return figure


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Render the configured knowledge network")
    parser.add_argument("--config", "-c", default="config.yaml")
    parser.add_argument("--output", "-o")
    parser.add_argument("--width", "-w", type=int, default=1200)
    parser.add_argument("--height", type=int, default=800)
    args = parser.parse_args(argv)
    NetworkVisualizer(args.config).visualize(args.output, args.width, args.height)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
