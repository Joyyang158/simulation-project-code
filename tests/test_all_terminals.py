import contextlib
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from airport_sim.experiments import run_experiments
from airport_sim.inputs import write_csv,read_csv

class AllTerminalTests(unittest.TestCase):
    def test_each_terminal_runs_in_isolation_with_its_own_flights_and_gates(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            rows=[]
            for terminal,count,service in [('T4',2,5),('T5',3,10),('T8',1,2)]:
                rows += [dict(date='2025-07-01',id=f'{terminal}-{i}',arrival=0,service=service,
                              terminal=terminal,airline=terminal) for i in range(count)]
            write_csv(root/'input.csv',rows)
            (root/'config.json').write_text(json.dumps(dict(gates_per_terminal=1,shared_gates_per_terminal=1,service_mode='observed_day')))
            with patch('airport_sim.plots.experiment_plots'),patch('airport_sim.plots.terminal_policy_plots'),contextlib.redirect_stdout(io.StringIO()):
                meta=run_experiments(root/'out',replications=2,data_path=root/'input.csv',config_path=root/'config.json')
            self.assertEqual(set(meta['terminal_scenarios']),{'T4','T5','T8'})
            self.assertEqual(set(meta['crn_by_terminal']),{'T4','T5','T8'})
            summary=read_csv(root/'out/summary.csv')
            for terminal,expected in [('T4',2.5),('T5',10),('T8',0)]:
                name=meta['terminal_scenarios'][terminal]['pooled']
                r=next(r for r in summary if r['scenario']==name and r['metric']=='mean_wait')
                self.assertEqual(float(r['mean']),expected)
                self.assertEqual(int(r['n']),2)
                records=read_csv(root/'out/flight_records'/f'{name}.csv')
                self.assertTrue(all(r['terminal']==terminal and r['gate_terminal']==terminal for r in records))
                self.assertEqual({r['id'] for r in records},{r['id'] for r in rows if r['terminal']==terminal})

if __name__=='__main__':unittest.main()
