"""Run the full 5x5 synthetic scaling grid, separate from the main BTS case study."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from airport_sim.scaling import run_scaling
if __name__=='__main__':
    root=Path(__file__).resolve().parents[1]
    run_scaling(root/'data/processed/jfk_2025_07.csv',root/'results/scaling')
