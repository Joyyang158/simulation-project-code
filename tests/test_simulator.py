import math
import sys
import unittest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from airport_sim.simulator import Flight, Gate, simulate
from airport_sim.inputs import synthetic_day


class SimulatorTests(unittest.TestCase):
    def test_fcfs_hand_calculation_and_time_weighted_metrics(self):
        demand = [Flight("a", 0, 4), Flight("b", 1, 2), Flight("c", 2, 1)]
        result = simulate(demand, [Gate("g", "T1")], horizon=5)
        self.assertEqual([r["wait"] for r in result.flights], [0, 3, 4])
        self.assertAlmostEqual(result.metrics["mean_queue"], 6 / 5)
        self.assertEqual(result.metrics["utilization"], 1)
        self.assertEqual(result.metrics["completed_in_day"], 1)
        self.assertEqual(result.metrics["completed_after_drain"], 3)

    def test_pooling_releases_head_of_line_block(self):
        flights = [Flight("a", 0, 10), Flight("b", 0, 1), Flight("c", 1, 1)]
        gates = [Gate("1", "T1"), Gate("2", "T1")]
        fixed = simulate(flights, gates, "dedicated", horizon=20)
        pooled = simulate(flights, gates, "pooled", horizon=20)
        self.assertEqual(fixed.flights[2]["wait"], 9)
        self.assertEqual(pooled.flights[2]["wait"], 0)

    def test_transfer_reserves_gate_and_counts_total_wait(self):
        gates = [Gate("a", "T1", initial_busy_until=20), Gate("b", "T2")]
        flights = [Flight("f1", 0, 5), Flight("f2", 1, 5)]
        r = simulate(flights, gates, "sharing", 30, 3)
        self.assertEqual([f["start"] for f in r.flights], [3, 11])
        self.assertEqual([f["wait"] for f in r.flights], [3, 10])
        self.assertEqual([f["queue_wait"] for f in r.flights], [0, 7])
        self.assertEqual(r.metrics["reserved_utilization"], 36 / 60)

    def test_compatibility_prevents_overflow(self):
        gates = [Gate("a", "T1", ("A",), initial_busy_until=20), Gate("b", "T2", ("B",))]
        r = simulate([Flight("f", 0, 5)], gates, "sharing", 30, 3)
        self.assertEqual(r.flights[0]["start"], 20)
        self.assertEqual(r.flights[0]["transfer"], 0)

    def test_closure_nonpreemptive_and_reopening(self):
        gate = Gate("g", "T1", closures=((2, 10),))
        r = simulate([Flight("a", 0, 5), Flight("b", 1, 1)], [gate], horizon=20)
        self.assertEqual([f["start"] for f in r.flights], [0, 10])
        self.assertEqual(r.metrics["utilization"], .3)

    def test_no_reservation_across_transfer_closure(self):
        gates = [Gate("a", "T1", initial_busy_until=20), Gate("b", "T2", closures=((2, 10),))]
        r = simulate([Flight("f", 0, 1)], gates, "sharing", 30, 3)
        self.assertEqual(r.flights[0]["start"], 13)

    def test_simultaneous_release_and_arrival(self):
        r = simulate([Flight("a", 0, 1), Flight("b", 1, 1)], [Gate("g", "T1")], horizon=2)
        self.assertEqual(r.flights[1]["wait"], 0)
        self.assertEqual(r.metrics["completed_in_day"], 2)

    def test_randomized_invariants_and_reproducibility(self):
        gates = [Gate("1", "T1"), Gate("2", "T1"), Gate("3", "T2", closures=((200, 350),)), Gate("4", "T2")]
        for seed in range(8):
            flights = synthetic_day(seed, 0, count=70)
            for policy in ("dedicated", "pooled", "sharing"):
                result = simulate(flights, gates, policy, transfer_time=10)
                self.assertEqual(len(result.flights), len(flights))
                for row in result.flights:
                    self.assertGreaterEqual(row["queue_wait"], 0)
                    self.assertAlmostEqual(row["wait"], row["queue_wait"] + row["transfer"])
                for g in gates:
                    spans = sorted([i for i in result.intervals if i["gate"] == g.id], key=lambda i: i["reserved"])
                    for a, b in zip(spans, spans[1:]):
                        self.assertLessEqual(a["end"], b["reserved"])
                self.assertEqual(result.metrics, simulate(flights, gates, policy, transfer_time=10).metrics)

    def test_single_gate_matches_lindley_recursion(self):
        flights = synthetic_day(9, 2, count=100)
        flights = [Flight(f.id, f.arrival, f.service) for f in flights]
        result = simulate(flights, [Gate("g", "T1")])
        end = 0
        for f, row in zip(flights, result.flights):
            start = max(f.arrival, end)
            self.assertAlmostEqual(row["start"], start)
            end = start + f.service

    def test_invalid_inputs_and_empty_day(self):
        g = [Gate("g", "T1")]
        for flight in (Flight("x", -1, 1), Flight("x", 0, 0), Flight("x", 0, math.inf), Flight("x", 0, 1, "missing")):
            with self.assertRaises(ValueError):
                simulate([flight], g)
        with self.assertRaises(ValueError):
            simulate([Flight("x", 0, 1)] * 2, g)
        empty = simulate([], g)
        self.assertEqual(empty.metrics["mean_queue"], 0)
        self.assertEqual(empty.metrics["completed_after_drain"], 0)


if __name__ == "__main__":
    unittest.main()
