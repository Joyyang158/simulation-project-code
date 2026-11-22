"""BTS adapter, explicit timing assumptions, and Ch. 5 input diagnostics."""
from collections import Counter, defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo
import csv
import json
import math
import numpy as np
from scipy import stats
from .random_inputs import erlang, streams
from .simulator import Flight


def iter_csv(path):
    with open(path, newline="", encoding="utf-8-sig") as handle:
        yield from csv.DictReader(handle)


def read_csv(path):
    return list(iter_csv(path))


def write_csv(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = list(rows)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def synthetic_day(seed, replication, horizon=1440, mean_service=45, count=200):
    """Illustrative two-terminal schedule, NOT a fitted airport dataset.

    Ch. 4 composition: choose a peak/off-peak component, then uniform time.
    Airline A's morning peak and B's afternoon peak create uneven terminal loads.
    """
    arrival_rng, service_rng, routing_rng = streams(seed, replication)
    arrivals = []
    terminal = routing_rng.choice(["T1", "T2"], size=count, p=[.65, .35])
    services = erlang(service_rng, mean_service, 4, count)
    for i, home in enumerate(terminal):
        if arrival_rng.random() < .65:
            low, high = (.25, .5) if home == "T1" else (.55, .8)
        else:
            low, high = .05, .95
        time = float(arrival_rng.uniform(low * horizon, high * horizon))
        arrivals.append(Flight(f"F{i:05d}", time, float(services[i]), home,
                               "A" if home == "T1" else "B"))
    return sorted(arrivals, key=lambda f: (f.arrival, f.id))


def load_days(path, horizon=1440):
    days = defaultdict(list)
    for row in read_csv(path):
        flight = Flight(row["id"], float(row["arrival"]), float(row["service"]),
                        row["terminal"], row["airline"])
        if not (0 <= flight.arrival < horizon and math.isfinite(flight.service) and flight.service > 0):
            raise ValueError(f"Invalid cleaned flight {flight.id}")
        days[row["date"]].append(flight)
    if not days:
        raise ValueError("Cleaned input contains no flights")
    for key in days:
        if len({f.id for f in days[key]}) != len(days[key]):
            raise ValueError(f"Duplicate flight IDs on {key}")
        days[key].sort(key=lambda f: (f.arrival, f.id))
    return dict(sorted(days.items()))


ALIASES = {
    "FlightDate": "FL_DATE", "Reporting_Airline": "OP_UNIQUE_CARRIER",
    "Tail_Number": "TAIL_NUM", "Flight_Number_Reporting_Airline": "OP_CARRIER_FL_NUM",
    "CRSDepTime": "CRS_DEP_TIME", "DepDelay": "DEP_DELAY",
    "ActualElapsedTime": "ACTUAL_ELAPSED_TIME", "TaxiIn": "TAXI_IN",
}


def clean_bts(raw_path, timezone_path, terminal_path, airport, output,
              baseline_taxi=10.0, clearance=5.0, max_stay=360.0):
    """Use scheduled local departure + delay, UTC elapsed time, then destination TZ.

    Requires domestic raw records for all observed legs of each tail, including
    adjacent days. Pair only consecutive observed legs; unobserved flights/towing
    remain limitations. Missing/nonpositive/overnight/long proxies are excluded.
    Terminal mapping is supplied by the user, never inferred as historical fact.
    """
    if not all(map(math.isfinite, (baseline_taxi, clearance, max_stay))) or baseline_taxi < 0 or clearance < 0 or max_stay <= 0:
        raise ValueError("Invalid cleaning parameters")
    tz_map = {r["airport"]: ZoneInfo(r["timezone"]) for r in read_csv(timezone_path)}
    terminal_map = {r["airline"]: r["terminal"] for r in read_csv(terminal_path)}
    if airport not in tz_map:
        raise ValueError("Target airport missing from timezone mapping")
    counts, tails = Counter(), defaultdict(list)
    seen = set()
    for raw in iter_csv(raw_path):
        counts["raw_rows"] += 1
        def get(name):
            return (raw.get(name, raw.get(ALIASES.get(name, name.upper()), "")) or "").strip()
        try:
            if float(get("Cancelled") or 0) or float(get("Diverted") or 0):
                counts["cancelled_or_diverted"] += 1
                continue
            origin, dest, tail = get("Origin"), get("Dest"), get("Tail_Number")
            if not tail:
                counts["missing_tail"] += 1
                continue
            if origin not in tz_map or dest not in tz_map:
                # Keep a tail-level flag: do not bridge a known unparsed leg.
                tails[tail].append(None)
                counts["missing_timezone"] += 1
                continue
            date = datetime.strptime(get("FlightDate")[:10], "%Y-%m-%d")
            hhmm = int(float(get("CRSDepTime")))
            if hhmm < 0 or hhmm > 2400 or hhmm % 100 >= 60:
                raise ValueError("Invalid HHMM")
            scheduled = (date + timedelta(hours=hhmm // 100, minutes=hhmm % 100)).replace(tzinfo=tz_map[origin])
            # Reject DST ambiguous/nonexistent local clock values.
            if scheduled.utcoffset() != scheduled.replace(fold=1).utcoffset():
                raise ValueError("Ambiguous departure local time")
            delay, elapsed = float(get("DepDelay")), float(get("ActualElapsedTime"))
            if not all(map(math.isfinite, (delay, elapsed))) or elapsed <= 0:
                raise ValueError("Invalid delay or elapsed time")
            dep = scheduled.astimezone(timezone.utc) + timedelta(minutes=delay)
            arr = dep + timedelta(minutes=elapsed)
            key = (tail, origin, dest, dep)
            if key in seen:
                counts["duplicates"] += 1
                continue
            seen.add(key)
            taxi = float(get("TaxiIn")) if get("TaxiIn") else None
            tails[tail].append(dict(dep=dep, arr=arr, origin=origin, dest=dest,
                                    taxi=taxi, airline=get("Reporting_Airline"), tail=tail))
            counts["parsed_legs"] += 1
        except (ValueError, KeyError, OverflowError):
            if get("Tail_Number"):
                tails[get("Tail_Number")].append(None)
            counts["invalid_time_or_missing_field"] += 1
    rows = []
    for tail, legs in tails.items():
        if None in legs:
            counts["tails_excluded_for_unparsed_legs"] += 1
            continue
        legs.sort(key=lambda r: r["dep"])
        for index, incoming in enumerate(legs):
            if incoming["dest"] != airport:
                continue
            counts["candidate_arrivals"] += 1
            if index + 1 == len(legs) or legs[index + 1]["origin"] != airport:
                counts["no_consecutive_departure"] += 1
                continue
            outgoing = legs[index + 1]
            stay = (outgoing["dep"] - incoming["arr"]).total_seconds() / 60
            local_arr = incoming["arr"].astimezone(tz_map[airport])
            local_dep = outgoing["dep"].astimezone(tz_map[airport])
            if local_arr.date() != local_dep.date():
                counts["overnight"] += 1
                continue
            if stay <= 0 or stay > max_stay:
                counts["invalid_or_long_stay"] += 1
                continue
            taxi = incoming["taxi"]
            if taxi is None or not math.isfinite(taxi) or taxi < 0:
                counts["invalid_taxi"] += 1
                continue
            airline = incoming["airline"]
            if airline not in terminal_map:
                counts["missing_terminal"] += 1
                continue
            request = (incoming["arr"] + timedelta(minutes=baseline_taxi - taxi)).astimezone(tz_map[airport])
            midnight = request.replace(hour=0, minute=0, second=0, microsecond=0)
            next_midnight = midnight + timedelta(days=1)
            if (next_midnight.astimezone(timezone.utc) - midnight.astimezone(timezone.utc)).total_seconds() != 86400:
                counts["dst_transition_day"] += 1
                continue
            arrival = (request.astimezone(timezone.utc) - midnight.astimezone(timezone.utc)).total_seconds() / 60
            rows.append(dict(date=str(request.date()), id=f"{tail}-{incoming['dep'].isoformat()}",
                             arrival=arrival, service=stay + clearance,
                             terminal=terminal_map[airline], airline=airline,
                             turnaround_proxy=stay))
    rows.sort(key=lambda r: (r["date"], r["arrival"], r["id"]))
    counts["retained_visits"] = len(rows)
    write_csv(output, rows)
    audit = dict(counts=dict(counts), airport=airport, baseline_taxi=baseline_taxi,
                 clearance=clearance, max_stay=max_stay,
                 limitations=["Retained matched domestic visits only; not full airport demand.",
                              "Unobserved flights and towing can invalidate occupancy proxies.",
                              "Terminal mapping and base taxi time are supplied assumptions."])
    Path(output).with_suffix(".audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    return audit


def diagnose_inputs(cleaned_path, output):
    """Fit on first 80% of dates; reserve remaining dates for inspection.

    Fitted-parameter KS statistics are descriptive only (no naive p-values).
    """
    from .plots import input_diagnostics
    days = load_days(cleaned_path)
    dates = list(days)
    split = max(1, min(len(dates) - 1, int(.8 * len(dates))))
    train_dates, holdout_dates = dates[:split], dates[split:]
    values = np.asarray([f.service for d in train_dates for f in days[d]])
    if len(values) < 5 or np.std(values) == 0:
        raise ValueError("Input fitting requires at least five nonconstant observations")
    holdout = np.asarray([f.service for d in holdout_dates for f in days[d]])
    fits = []
    for name in ("expon", "gamma", "lognorm"):
        distribution = getattr(stats, name)
        parameters = distribution.fit(values, floc=0)
        ks = stats.kstest(values, distribution.cdf, args=parameters).statistic
        fits.append(dict(distribution=name, parameters=list(map(float, parameters)),
                         ks_statistic=float(ks),
                         aic=float(2 * (len(parameters) - 1) - 2 * distribution.logpdf(values, *parameters).sum()),
                         holdout_ks_statistic=float(stats.kstest(holdout, distribution.cdf, args=parameters).statistic) if len(holdout) else None))
    correlations = []
    for d in train_dates:
        v = np.asarray([f.service for f in days[d]])
        if len(v) >= 4 and np.std(v[:-1]) > 0 and np.std(v[1:]) > 0:
            correlations.append(dict(date=d, lag1=float(np.corrcoef(v[:-1], v[1:])[0, 1])))
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    summary = dict(training_dates=train_dates, holdout_dates=holdout_dates,
                   n=len(values), mean=float(values.mean()), cv=float(values.std(ddof=1) / values.mean()),
                   fits=fits, daily_lag1=correlations,
                   note="KS statistics are descriptive: parameters estimated from the same data. No automatic fit acceptance.")
    (output / "input_fit.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    input_diagnostics(values, fits, days, train_dates, holdout_dates, output)
    return summary
