"""Demand scaling must preserve source counts, capacity and one-server controls."""
import contextlib
import io
import json
from collections import Counter
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from airport_sim.experiments import expand_day, run_experiments, configuration
from airport_sim.inputs import write_csv, read_csv
from airport_sim.simulator import Flight


class DemandStressTests(unittest.TestCase):
    def test_expansion_preserves_bins_counts_originals_and_nested_inputs(self):
        flights = [Flight('a', 0, 70, 'T4', 'DL'), Flight('b', 1439.5, 80, 'T5', 'B6'),
                   Flight('a::stress1', 610, 90, 'T4', 'DL')]
        a = expand_day(flights, 4, np.random.default_rng(3), 1440)
        b = expand_day(flights, 5, np.random.default_rng(3), 1440)
        self.assertEqual(a, b[:len(a)])
        self.assertEqual(b[:3], flights)
        self.assertEqual(len(set(f.id for f in b)), 15)
        expected = Counter((f.airline, f.terminal, int(f.arrival // 30)) for f in flights)
        actual = Counter((f.airline, f.terminal, int(f.arrival // 30)) for f in b)
        self.assertEqual(actual, Counter({k: 5*v for k,v in expected.items()}))
        self.assertTrue(all(0 <= f.arrival < 1440 for f in b))

    def test_stress_modes_align_counts_and_leave_single_gate_unchanged(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root/'input.csv'
            write_csv(source, [dict(date=f'2025-07-{d+1:02d}', id=f'f{i}', arrival=100+i*5,
                service=70+i, terminal=['T4','T5','T8'][i%3], airline=['DL','B6','AA'][i%3])
                for d in range(2) for i in range(9+d)])
            outputs = {}
            with patch('airport_sim.plots.experiment_plots'), contextlib.redirect_stdout(io.StringIO()):
                for mode,m in [('empirical',1),('empirical',5),('calibrated_synthetic',5)]:
                    name=f'{mode}_{m}'
                    cfg=root/f'{name}.json'
                    cfg.write_text(json.dumps(dict(input_mode=mode,demand_multiplier=m,
                        gates_per_terminal=20,shared_gates_per_terminal=10)))
                    metadata=run_experiments(root/name,replications=2,data_path=source,config_path=cfg)
                    self.assertEqual(len(metadata['gates']),60)
                    outputs[name]=read_csv(root/name/'replications.csv')
            base=outputs['empirical_1']; stress=outputs['empirical_5']; syn=outputs['calibrated_synthetic_5']
            for a,b,c in zip(base,stress,syn):
                self.assertEqual(b['sampled_date'],c['sampled_date'])
                self.assertEqual(b['n_flights'],c['n_flights'])
                if a['scenario'].startswith('single_'):
                    self.assertEqual(a,b)
                else:
                    self.assertEqual(float(b['n_flights']),5*float(a['n_flights']))
            for mode in ['empirical_5','calibrated_synthetic_5']:
                rows=read_csv(root/mode/'flight_records/network_no_sharing.csv')
                self.assertEqual(len({r['id'] for r in rows}),len(rows))
            cfg=root/'bad.json'
            for value in [0,-1,1.5,True]:
                cfg.write_text(json.dumps(dict(demand_multiplier=value)))
                with self.assertRaises(ValueError): configuration(cfg)


if __name__ == '__main__': unittest.main()
