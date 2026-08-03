---
title: Model Context Protocol
type: tool
status: stable
created: 2026-08-02
tags:
  - mcp
  - model_context_protocol
  - tools
semantic_relations:
  - type: documents
    links:
      - '[[../AGENTS|Tools documentation instructions]]'
---

# Model Context Protocol

The Model Context Protocol (MCP) is an open standard published by Anthropic
for connecting AI applications to external tools and data sources
(see the specification at modelcontextprotocol.io). It is a general
integration standard, not a repository component.

## Status in this repository

This repository does not ship an MCP server or MCP client code. Its
AI-assistant integration surface is:

- `.cursorrules`, `AGENTS.md`, `CLAUDE.md` — guidance for AI-assisted
  editors.
- The installed CLI entry points (`cognitive-*`) — scriptable interfaces.
- The documentation validators — machine-readable JSON reports
  (`--json` flags on `cognitive-validate-docs`, `cognitive-verify-links`).

If an MCP server is added later, it belongs in `code/tools/src` with a
regression test and documentation here. Until then, treat any MCP-related
content as external reference material.
