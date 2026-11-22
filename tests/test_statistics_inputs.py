import json
from pathlib import Path
import sys
import tempfile
import unittest
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import numpy as np
from airport_sim.inputs import clean_bts, load_days, write_csv, synthetic_day
from airport_sim.metrics import confidence_interval, paired_difference
from airport_sim.random_inputs import streams, exponential, erlang
from airport_sim.experiments import smooth_day


class StatisticsTests(unittest.TestCase):
    def test_student_t_interval_known_example(self):
        ci = confidence_interval([1, 2, 3, 4, 5])
        self.assertAlmostEqual(ci["mean"], 3)
        self.assertAlmostEqual(ci["ci_high"], 4.963243161, places=7)
        self.assertEqual(paired_difference([1, 2, 3], [2, 3, 4])["sd"], 0)

    def test_random_variables_have_expected_moments(self):
        a, b, _ = streams(553, 0)
        exp = exponential(a, 10, 100000)
        erl = erlang(b, 10, 4, 100000)
        self.assertAlmostEqual(float(exp.mean()), 10, delta=.12)
        self.assertAlmostEqual(float(erl.mean()), 10, delta=.08)
        self.assertAlmostEqual(float(erl.std() / erl.mean()), .5, delta=.01)
        np.testing.assert_array_equal(streams(1, 3)[0].random(10), streams(1, 3)[0].random(10))
        self.assertFalse(np.array_equal(streams(1, 3)[0].random(10), streams(1, 4)[0].random(10)))

    def test_smoothing_preserves_flights_and_service_inputs(self):
        flights = synthetic_day(1, 0)
        smooth = smooth_day(flights, 1440, .5)
        self.assertEqual([(f.id, f.service) for f in smooth], [(f.id, f.service) for f in flights])
        self.assertTrue(all(0 <= f.arrival < 1440 for f in smooth))
        self.assertTrue(all(a.arrival <= b.arrival for a, b in zip(smooth, smooth[1:])))


class BTSAdapterTests(unittest.TestCase):
    def test_cross_midnight_and_timezone_correct_pair(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp)
            write_csv(p / "tz.csv", [dict(airport="LAX", timezone="America/Los_Angeles"), dict(airport="JFK", timezone="America/New_York")])
            write_csv(p / "term.csv", [dict(airline="AA", terminal="T1")])
            # Jan 1 23:00 PST + 300 min = Jan 2 07:00 EST; departs 08:00 EST.
            legs = [dict(FL_DATE="2025-01-01", ORIGIN="LAX", DEST="JFK", TAIL_NUM="N1", OP_UNIQUE_CARRIER="AA",
                         CRS_DEP_TIME=2300, DEP_DELAY=0, ACTUAL_ELAPSED_TIME=300, TAXI_IN=15, CANCELLED=0, DIVERTED=0),
                    dict(FL_DATE="2025-01-02", ORIGIN="JFK", DEST="LAX", TAIL_NUM="N1", OP_UNIQUE_CARRIER="AA",
                         CRS_DEP_TIME=800, DEP_DELAY=0, ACTUAL_ELAPSED_TIME=360, TAXI_IN=10, CANCELLED=0, DIVERTED=0)]
            write_csv(p / "raw.csv", legs + [legs[0]])
            audit = clean_bts(p / "raw.csv", p / "tz.csv", p / "term.csv", "JFK", p / "visits.csv")
            self.assertEqual(audit["counts"]["duplicates"], 1)
            visits = load_days(p / "visits.csv")
            f = visits["2025-01-02"][0]
            self.assertEqual(f.arrival, 415)  # 07:00 - 15 + assumed 10 min.
            self.assertEqual(f.service, 65)  # 60 min proxy + 5 min buffer.
            # A known but unparseable intervening leg must not be bridged.
            bad = dict(legs[0], ORIGIN="XXX", CRS_DEP_TIME=2200)
            write_csv(p / "raw.csv", legs + [bad])
            audit = clean_bts(p / "raw.csv", p / "tz.csv", p / "term.csv", "JFK", p / "visits.csv")
            self.assertEqual(audit["counts"]["retained_visits"], 0)
            self.assertEqual(audit["counts"]["tails_excluded_for_unparsed_legs"], 1)


if __name__ == "__main__":
    unittest.main()
