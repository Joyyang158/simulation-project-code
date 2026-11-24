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
            self.assertEqual(len(replications), 48)
            self.assertTrue((root / "output" / "experiment_overview.pdf").is_file())
            diff = read_csv(root / "output" / "paired_differences.csv")
            self.assertTrue(all(float(r["mean"]) == 0 for r in diff if r["metric"] == "n_flights"))
            self.assertTrue(json.loads((root / "output" / "metadata.json").read_text())["packages"])

    def test_calibrated_inputs_keep_counts_capacity_and_reproducibility(self):
        from unittest.mock import patch
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "visits.csv"
            write_csv(source, [dict(date=f"2025-07-{day+1:02d}", id=f.id,
                arrival=f.arrival, service=f.service, terminal=f.terminal, airline=f.airline)
                for day in range(3) for f in synthetic_day(7, day, count=20+day)])
            config = root / "config.json"
            config.write_text(json.dumps(dict(input_mode="calibrated_synthetic",
                gates_per_terminal=20, shared_gates_per_terminal=10)))
            with patch("airport_sim.plots.experiment_plots"), contextlib.redirect_stdout(io.StringIO()):
                first = run_experiments(root / "a", replications=2, data_path=source, config_path=config)
                run_experiments(root / "b", replications=2, data_path=source, config_path=config)
                config.write_text(json.dumps(dict(gates_per_terminal=20, shared_gates_per_terminal=10)))
                real = run_experiments(root / "real", replications=2, data_path=source, config_path=config)
            self.assertEqual((root / "a/replications.csv").read_bytes(), (root / "b/replications.csv").read_bytes())
            self.assertEqual(first["gates"], real["gates"])
            a = read_csv(root / "a/replications.csv")
            b = read_csv(root / "real/replications.csv")
            self.assertEqual([(r["sampled_date"], r["n_flights"]) for r in a],
                             [(r["sampled_date"], r["n_flights"]) for r in b])
            generated = read_csv(root / "a/flight_records/network_no_sharing.csv")
            observed = read_csv(root / "real/flight_records/network_no_sharing.csv")
            self.assertEqual({r["id"] for r in generated}, {r["id"] for r in observed})
            self.assertTrue(all(0 <= float(r["arrival"]) < 1440 and float(r["service"]) > 0 for r in generated))
            self.assertNotEqual([r["arrival"] for r in generated], [r["arrival"] for r in observed])

    def test_observed_day_mode_preserves_service_inputs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            flights = synthetic_day(3, 0, count=12)
            write_csv(root / "visits.csv", [dict(date="2025-07-01", id=f.id,
                arrival=f.arrival, service=f.service, terminal=f.terminal, airline=f.airline) for f in flights])
            (root / "config.json").write_text(json.dumps(dict(service_mode="observed_day", single_input_mode="sampled_day")))
            with contextlib.redirect_stdout(io.StringIO()):
                metadata = run_experiments(root / "output", replications=2,
                    config_path=root / "config.json", data_path=root / "visits.csv")
            original = {f.id: f.service for f in flights}
            records = read_csv(root / "output/flight_records/network_no_sharing.csv")
            self.assertEqual({r["id"]: float(r["service"]) for r in records}, original)
            self.assertIn("Observed day", metadata["service_sampling"])
            import numpy as np
            for terminal in {f.terminal for f in flights}:
                baseline = read_csv(root / f"output/flight_records/single_{terminal}_base.csv")
                reduced = read_csv(root / f"output/flight_records/single_{terminal}_low_variability.csv")
                expected = {f.id: (f.arrival, f.service) for f in flights if f.terminal == terminal}
                self.assertEqual({r['id']: (float(r['arrival']), float(r['service'])) for r in baseline}, expected)
                self.assertEqual({r['id']: float(r['arrival']) for r in reduced}, {k: v[0] for k,v in expected.items()})
                original_services = np.array([float(r['service']) for r in baseline])
                reduced_services = np.array([float(r['service']) for r in reduced])
                self.assertAlmostEqual(original_services.mean(), reduced_services.mean())
                self.assertAlmostEqual(original_services.var()/4, reduced_services.var())
            scenarios = {r['scenario'] for r in read_csv(root / 'output/replications.csv')}
            self.assertNotIn('single_0.5_base', scenarios)


if __name__ == "__main__":
    unittest.main()
