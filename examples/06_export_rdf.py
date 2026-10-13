"""Export the knowledge base as RDF (N-Quads and Turtle) for the Open Knowledge Network."""

import argparse
from pathlib import Path

from scripts.export_rdf import main

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--output", type=Path, default=Path("output/examples/06_rdf"))
parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parent.parent)
args = parser.parse_args()
raise SystemExit(main([str(args.root), "--output", str(args.output)]))
