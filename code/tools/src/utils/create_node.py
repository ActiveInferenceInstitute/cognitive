"""Create knowledge-base nodes from the repository's documented templates."""

from __future__ import annotations

import re
from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import click
import yaml
from jinja2 import Environment, FileSystemLoader, TemplateNotFound


class NodeCreator:
    """Render a node template using paths relative to its configuration file."""

    _PLURAL_TYPES = {
        "agent": "agents",
        "belief": "beliefs",
        "goal": "goals",
        "action": "actions",
        "observation": "observations",
        "environment": "environments",
    }

    def __init__(self, config_path: str | Path = "config.yaml") -> None:
        self.config_path = Path(config_path).expanduser().resolve()
        self.config = self._load_config(self.config_path)
        paths = self.config.get("paths")
        if not isinstance(paths, dict):
            raise ValueError("Configuration must contain a paths mapping")
        self.template_path = self._resolve_path(paths, "templates")
        self.knowledge_base_path = self._resolve_path(paths, "knowledge_base")
        self.env = Environment(
            loader=FileSystemLoader(self.template_path),
            trim_blocks=True,
            lstrip_blocks=True,
            autoescape=False,
        )

    @staticmethod
    def _load_config(config_path: Path) -> dict[str, Any]:
        try:
            with config_path.open(encoding="utf-8") as config_file:
                config = yaml.safe_load(config_file) or {}
        except OSError as exc:
            raise ValueError(f"Unable to read configuration {config_path}: {exc}") from exc
        if not isinstance(config, dict):
            raise ValueError("Configuration must be a YAML mapping")
        return config

    def _resolve_path(self, paths: Mapping[str, Any], key: str) -> Path:
        value = paths.get(key)
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"paths.{key} must be a non-empty path")
        candidate = Path(value).expanduser()
        if not candidate.is_absolute():
            candidate = self.config_path.parent / candidate
        return candidate.resolve()

    @staticmethod
    def _safe_name(name: str) -> str:
        candidate = name.strip()
        if not candidate or candidate in {".", ".."}:
            raise ValueError("name must be non-empty")
        if Path(candidate).name != candidate or "/" in candidate or "\\" in candidate:
            raise ValueError("name must be a single filename, not a path")
        safe = re.sub(r"[^A-Za-z0-9._ -]", "_", candidate).strip(" .")
        if not safe:
            raise ValueError("name does not contain filename-safe characters")
        return safe

    @staticmethod
    def _generate_id(prefix: str) -> str:
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        safe_prefix = re.sub(r"[^A-Za-z0-9_-]", "_", prefix.strip())
        return f"{safe_prefix}_{timestamp}"

    def create_node(
        self,
        node_type: str,
        name: str,
        template_vars: Mapping[str, Any] | None = None,
    ) -> Path:
        """Render and write one node, returning its absolute path."""
        node_type = node_type.strip().lower()
        if not re.fullmatch(r"[a-z][a-z0-9_-]*", node_type):
            raise ValueError("node_type must contain lowercase letters, digits, '_' or '-'")
        safe_name = self._safe_name(name)
        template_name = f"{node_type}_template.md"
        try:
            template = self.env.get_template(template_name)
        except TemplateNotFound as exc:
            raise FileNotFoundError(
                f"No template for node type '{node_type}': expected "
                f"{self.template_path / template_name}"
            ) from exc

        variables = dict(template_vars or {})
        now = datetime.now(timezone.utc).isoformat()
        variables.update(
            {
                "node_id": self._generate_id(node_type),
                "node_name": safe_name,
                "date": now,
                "type": node_type,
            }
        )
        node_id = variables["node_id"]
        variables.setdefault(f"{node_type}_id", node_id)
        variables.setdefault(f"{node_type}_name", safe_name)
        content = template.render(**variables)
        output_dir = self.knowledge_base_path / self._PLURAL_TYPES.get(node_type, f"{node_type}s")
        output_dir.mkdir(parents=True, exist_ok=True)
        output_file = output_dir / f"{safe_name}.md"
        if output_file.exists():
            raise FileExistsError(f"Node already exists: {output_file}")
        output_file.write_text(content, encoding="utf-8")
        return output_file


@click.command()
@click.option("--type", "node_type", required=True, help="Template type to render")
@click.option("--name", required=True, help="Filename and display name")
@click.option("--config", default="config.yaml", show_default=True, help="YAML configuration path")
@click.option("--var", "variables", multiple=True, help="Template variable in key=value form")
def main(node_type: str, name: str, config: str, variables: tuple[str, ...]) -> None:
    """Create a knowledge-base node from a configured template."""
    template_vars: dict[str, str] = {}
    for variable in variables:
        if "=" not in variable:
            raise click.ClickException(f"Variable must use key=value syntax: {variable}")
        key, value = variable.split("=", 1)
        key = key.strip()
        if not key:
            raise click.ClickException("Variable names cannot be empty")
        template_vars[key] = value.strip()
    try:
        output_file = NodeCreator(config).create_node(node_type, name, template_vars)
    except (OSError, ValueError, FileExistsError) as exc:
        raise click.ClickException(str(exc)) from exc
    click.echo(str(output_file))


if __name__ == "__main__":
    main()
