import argparse
import json
from .experiments import run_experiments, validate_mm1
from .inputs import clean_bts, diagnose_inputs


def main():
    parser = argparse.ArgumentParser(description="Airport gate queue simulation (times in minutes)")
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run", help="Run all three experiment families")
    run.add_argument("--output", default="results/demo")
    run.add_argument("--replications", type=int, default=100)
    run.add_argument("--seed", type=int, default=553)
    run.add_argument("--config")
    run.add_argument("--data", help="Cleaned CSV; omit for labeled synthetic demonstration")
    validate = sub.add_parser("validate", help="M/M/1 steady-state benchmark")
    validate.add_argument("--output", default="results/validation")
    validate.add_argument("--replications", type=int, default=100)
    validate.add_argument("--seed", type=int, default=553)
    validate.add_argument("--warmup", type=float, default=5000)
    validate.add_argument("--duration", type=float, default=20000)
    clean = sub.add_parser("clean", help="Clean BTS records into simulation visits")
    clean.add_argument("--raw", required=True)
    clean.add_argument("--timezones", required=True, help="CSV: airport,timezone (domestic airports)")
    clean.add_argument("--terminals", required=True, help="CSV: airline,terminal")
    clean.add_argument("--airport", default="JFK")
    clean.add_argument("--output", default="data/processed/visits.csv")
    clean.add_argument("--baseline-taxi", type=float, default=10)
    clean.add_argument("--clearance", type=float, default=0)
    clean.add_argument("--max-stay", type=float, default=360)
    diagnose = sub.add_parser("diagnose", help="Ch. 5 input fit and diagnostic figures")
    diagnose.add_argument("--data", required=True)
    diagnose.add_argument("--output", default="results/inputs")
    args = parser.parse_args()
    if args.command == "run":
        result = run_experiments(args.output, args.replications, args.seed, args.config, args.data)
        print(json.dumps(dict(output=args.output, source=result["input_source"], crn=result["crn"]), indent=2))
    elif args.command == "validate":
        validate_mm1(args.output, args.replications, args.seed, args.warmup, args.duration)
        print(f"Validation results: {args.output}")
    elif args.command == "clean":
        print(json.dumps(clean_bts(args.raw, args.timezones, args.terminals, args.airport,
                                   args.output, args.baseline_taxi, args.clearance, args.max_stay), indent=2))
    else:
        result = diagnose_inputs(args.data, args.output)
        print(f"Input diagnostics: {args.output}; {result['n']} training observations")
