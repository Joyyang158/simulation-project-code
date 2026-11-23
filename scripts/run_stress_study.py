"""Reproduce the aligned 20-gate demand study from the cleaned BTS input."""
import argparse
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from airport_sim.experiments import run_experiments

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=['synthetic', 'bts', 'observed', 'all'], default='all')
    parser.add_argument('--replications', type=int, default=100)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    modes = ['synthetic', 'bts', 'observed'] if args.mode == 'all' else [args.mode]
    for mode in modes:
        for multiplier in ([5] if mode == 'observed' else [1, 4, 5, 6]):
            name = 'bts_20_x5_observed' if mode == 'observed' else f'{mode}_20_x{multiplier}'
            config = (f'aligned_{mode}_20.json' if multiplier == 1
                      else f'stress_{name}.json')
            run_experiments(root/'results/stress'/name, replications=args.replications,
                config_path=root/'configs'/config, data_path=root/'data/processed/jfk_2025_07.csv')
