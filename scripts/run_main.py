"""Reproduce both aligned four-gate main studies, without demand expansion."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from airport_sim.experiments import run_experiments
if __name__=='__main__':
    root=Path(__file__).resolve().parents[1]
    for mode in ['synthetic','bts','bts_resampled']:
        run_experiments(root/'results/main'/mode,replications=100,seed=553,
            data_path=root/'data/processed/jfk_2025_07.csv',config_path=root/f'configs/main_{mode}_4.json')
