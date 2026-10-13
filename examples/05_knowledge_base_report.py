"""Measure and draw how the knowledge base is connected."""

import argparse
from pathlib import Path

from scripts.kb_report import main

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--output", type=Path, default=Path("output/examples/05_kb_report"))
parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parent.parent)
args = parser.parse_args()
raise SystemExit(main([str(args.root), "--output", str(args.output)]))
