import sys
from pathlib import Path
import unittest
from collections import Counter
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from airport_sim.scaling import synthetic_network, capacity
from airport_sim.simulator import simulate

class ScalingTests(unittest.TestCase):
    def test_nested_demand_and_capacity_grid(self):
        profile=dict(flights_per_terminal_day=10,bin_edges=[0,720,1440],arrival_probabilities=[.6,.4],service_mean=100,service_variance=2500)
        full=synthetic_network(profile,5,553,0)
        self.assertEqual(full,synthetic_network(profile,5,553,0))
        for n in range(1,6):
            flights=synthetic_network(profile,n,553,0)
            self.assertEqual(flights,[f for f in full if int(f.terminal[1:])<=n])
            self.assertEqual(len({f.id for f in flights}),len(flights))
            self.assertTrue(all(0<=f.arrival<1440 and f.service>0 for f in flights))
            for g in [1,5,10,20,50]:
                gates=capacity(n,g)
                self.assertEqual(Counter(x.terminal for x in gates),{f'T{t}':g for t in range(1,n+1)})
                if n>1:self.assertEqual(sum(len(x.airlines)==n for x in gates),n*((g+1)//2))
        flights=synthetic_network(profile,1,553,0)
        gates=capacity(1,5)
        self.assertEqual(simulate(flights,gates).metrics,simulate(flights,gates,'sharing',transfer_time=20).metrics)
