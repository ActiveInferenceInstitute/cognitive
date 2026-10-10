"""Locate fenced code blocks in Markdown with a line-based fence tracker.

A non-greedy regex such as ```` ```[\\s\\S]*?``` ```` cannot read Markdown
fences. It ends a block at the first three backticks it meets, so a
four-backtick fence that shows a three-backtick example closes early and the
text after the example is read as prose, and it ignores a fence that is never
closed, so everything after it is read as prose too. This module follows the
CommonMark rules that matter for tooling instead:

- an opening fence is a run of at least three backticks or tildes at the start
  of a line (after any indentation, blockquote markers and list markers, so a
  fence in a callout or on a list-item line is found); a backtick fence's info string may not
  contain a backtick, which keeps inline ```` ```code``` ```` from opening a block;
- a closing fence uses the same character, is at least as long as the opening
  fence, and has nothing after it but whitespace, so a shorter run or a run of
  the other character inside a block is content;
- a block that is never closed runs to the end of the file.

Indentation is not limited to three spaces, so fences nested inside list items
are found. The cost is that a fence-looking line inside an indented code block
is read as a fence, which is the safer error for a link or heading scanner.
"""

from __future__ import annotations

import re
from collections.abc import Iterator
from dataclasses import dataclass

# A fence may sit inside a blockquote or callout (``> ```), and an opening fence
# may follow a list marker on the same line (``- ```python``); a closing fence is
# always on its own line, optionally behind blockquote markers and indentation.
_QUOTE = r"(?:[ \t]*>)*"
_OPENING = re.compile(
    rf"^(?P<quote>{_QUOTE})(?P<lead>[ \t]*(?:(?:[-*+]|\d{{1,9}}[.)])[ \t]+)*)"
    r"(?P<fence>`{3,}|~{3,})(?P<info>[^\r\n]*)$"
)
_CLOSING = re.compile(rf"^{_QUOTE}[ \t]*(?P<fence>`{{3,}}|~{{3,}})[ \t]*$")
_QUOTE_PREFIX = re.compile(r"^[ \t]*>[ \t]?")
# CommonMark ends a line only at LF, CR or CRLF; str.splitlines also splits at
# U+2028, U+0085 and the like, which turns a stray character into a fake fence.
_LINE = re.compile(r"[^\r\n]*(?:\r\n|\r|\n)|[^\r\n]+")


@dataclass(frozen=True)
class FencedBlock:
    """One fenced code block.

    ``start_line`` and ``end_line`` are 1-based and inclusive of the fence
    lines. For an unterminated block ``end_line`` is the last line of the file
    and ``terminated`` is false. ``indent`` is the width of the opening fence's
    indentation, which is non-zero for a fence nested in a list item.
    """

    info: str
    body: str
    start_line: int
    end_line: int
    terminated: bool
    indent: int


def _scan(lines: list[str]) -> Iterator[tuple[int, int, str, bool]]:
    """Yield ``(first, last, info, terminated)`` as 0-based inclusive line indexes."""
    index = 0
    while index < len(lines):
        line = lines[index].rstrip("\r\n")
        opening = _OPENING.match(line)
        if opening is None or (opening["fence"][0] == "`" and "`" in opening["info"]):
            index += 1
            continue
        char, length = opening["fence"][0], len(opening["fence"])
        last = len(lines) - 1
        terminated = False
        for close in range(index + 1, len(lines)):
            closing = _CLOSING.match(lines[close].rstrip("\r\n"))
            if closing and closing["fence"][0] == char and len(closing["fence"]) >= length:
                last, terminated = close, True
                break
        yield index, last, opening["info"].strip(), terminated
        index = last + 1


def _dedent(line: str, width: int, quote_depth: int = 0) -> str:
    """Remove ``quote_depth`` blockquote markers, then up to ``width`` spaces or tabs."""
    for _ in range(quote_depth):
        line = _QUOTE_PREFIX.sub("", line, count=1)
    removable = len(line) - len(line.lstrip(" \t"))
    return line[min(width, removable) :]


def _split_lines(content: str) -> list[str]:
    """Split on LF, CR or CRLF only, keeping the line endings."""
    return _LINE.findall(content)


def iter_fenced_blocks(content: str) -> Iterator[FencedBlock]:
    """Yield every fenced code block in ``content`` in document order."""
    lines = _split_lines(content)
    for first, last, info, terminated in _scan(lines):
        body_end = last if terminated else last + 1
        # Content lines lose the blockquote markers and up to as much indentation
        # as the opening fence has, list marker included.
        opening = _OPENING.match(lines[first].rstrip("\r\n"))
        assert opening is not None  # _scan only yields lines that open a fence
        quote_depth = opening["quote"].count(">")
        indent = len(opening["lead"])
        body = "".join(_dedent(line, indent, quote_depth) for line in lines[first + 1 : body_end])
        yield FencedBlock(
            info=info,
            body=body.rstrip("\r\n"),
            start_line=first + 1,
            end_line=last + 1,
            terminated=terminated,
            indent=indent,
        )


def strip_fenced_blocks(content: str, replacement: str = "") -> str:
    """Return ``content`` without its fenced code blocks, fence lines included.

    ``replacement`` is emitted once in place of each removed block. Pass
    ``"\\n"`` to keep the prose on either side of a block in separate paragraphs.
    """
    lines = _split_lines(content)
    kept: list[str] = []
    cursor = 0
    for first, last, _, _ in _scan(lines):
        kept.extend(lines[cursor:first])
        kept.append(replacement)
        cursor = last + 1
    kept.extend(lines[cursor:])
    return "".join(kept)
