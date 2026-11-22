"""Exercise the CSV-to-experiments interface using explicitly synthetic fixtures."""
import contextlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from airport_sim.inputs import synthetic_day, write_csv, read_csv, diagnose_inputs
from airport_sim.experiments import run_experiments


class PipelineTests(unittest.TestCase):
    def test_cleaned_csv_diagnostics_and_experiment_outputs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            rows = []
            for day in range(5):
                for f in synthetic_day(7, day, count=20):
                    rows.append(dict(date=f"2025-01-{day + 1:02d}", id=f.id,
                                     arrival=f.arrival, service=f.service,
                                     terminal=f.terminal, airline=f.airline))
            write_csv(root / "visits.csv", rows)
            fitted = diagnose_inputs(root / "visits.csv", root / "diagnostics")
            self.assertEqual(len(fitted["training_dates"]), 4)
            self.assertEqual(len(fitted["holdout_dates"]), 1)
            self.assertTrue(all(r["holdout_ks_statistic"] is not None for r in fitted["fits"]))
            with contextlib.redirect_stdout(io.StringIO()):
                metadata = run_experiments(root / "output", replications=2, data_path=root / "visits.csv")
            self.assertEqual(metadata["replications"], 2)
            self.assertEqual(len(metadata["input_sha256"]), 64)
            replications = read_csv(root / "output" / "replications.csv")
            self.assertEqual(len(replications), 40)
            self.assertTrue((root / "output" / "experiment_overview.pdf").is_file())
            diff = read_csv(root / "output" / "paired_differences.csv")
            self.assertTrue(all(float(r["mean"]) == 0 for r in diff if r["metric"] == "n_flights"))
            self.assertTrue(json.loads((root / "output" / "metadata.json").read_text())["packages"])

    def test_observed_day_mode_preserves_service_inputs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            flights = synthetic_day(3, 0, count=12)
            write_csv(root / "visits.csv", [dict(date="2025-07-01", id=f.id,
                arrival=f.arrival, service=f.service, terminal=f.terminal, airline=f.airline) for f in flights])
            (root / "config.json").write_text(json.dumps(dict(service_mode="observed_day")))
            with contextlib.redirect_stdout(io.StringIO()):
                metadata = run_experiments(root / "output", replications=2,
                    config_path=root / "config.json", data_path=root / "visits.csv")
            original = {f.id: f.service for f in flights}
            records = read_csv(root / "output/flight_records/network_no_sharing.csv")
            self.assertEqual({r["id"]: float(r["service"]) for r in records}, original)
            self.assertIn("Observed day", metadata["service_sampling"])


if __name__ == "__main__":
    unittest.main()
