"""Convenient source-tree entry point: python run.py run --replications 30."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
from airport_sim.cli import main

if __name__ == "__main__":
    main()
