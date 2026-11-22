"""Prepare official BTS archives for a reproducible JFK July 2025 case study.

Run from the project root after downloading data/raw/bts/2025_{6,7,8}.zip
and data/reference/openflights_airports.dat. This script does not infer real
gate capacity from the airport's total gate counts.
"""
from collections import Counter
from datetime import datetime, timedelta, timezone
from pathlib import Path
import csv
import hashlib
import io
import json
import sys
import zipfile
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from airport_sim.inputs import clean_bts, read_csv, write_csv


def sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main():
    rawdir = ROOT / "data/raw/bts"
    reference = ROOT / "data/reference"
    processed = ROOT / "data/processed"
    processed.mkdir(parents=True, exist_ok=True)
    fields = ["FlightDate", "Reporting_Airline", "Tail_Number", "Origin", "Dest",
              "CRSDepTime", "DepDelay", "ActualElapsedTime", "TaxiIn", "Cancelled",
              "Diverted", "DepTime", "ArrTime", "WheelsOn", "Flight_Number_Reporting_Airline"]
    selected = {"AA", "B6", "DL"}
    counts, arrival_counts, airports = Counter(), Counter(), set()
    manifest = dict(created_utc=datetime.now(timezone.utc).isoformat(),
                    target_month="2025-07", context_dates=["2025-06-30", "2025-08-01"],
                    airport="JFK", carriers=sorted(selected), sources=[])
    rawcsv = rawdir / "selected_carriers_context.csv"
    with rawcsv.open("w", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=fields)
        writer.writeheader()
        for month in (6, 7, 8):
            path = rawdir / f"2025_{month}.zip"
            with zipfile.ZipFile(path) as archive:
                bad = archive.testzip()
                if bad:
                    raise ValueError(f"CRC validation failed: {path}: {bad}")
                members = [n for n in archive.namelist() if n.lower().endswith(".csv")]
                if len(members) != 1:
                    raise ValueError(f"Expected one CSV in {path}, got {members}")
                manifest["sources"].append(dict(url=f"https://www.transtats.bts.gov/PREZIP/On_Time_Reporting_Carrier_On_Time_Performance_1987_present_2025_{month}.zip",
                                                 local=str(path.relative_to(ROOT)), bytes=path.stat().st_size,
                                                 sha256=sha256(path), csv_member=members[0]))
                with archive.open(members[0]) as handle:
                    reader = csv.DictReader(io.TextIOWrapper(handle, encoding="utf-8-sig"))
                    missing = set(fields) - set(reader.fieldnames)
                    if missing:
                        raise ValueError(f"Missing columns: {missing}")
                    for row in reader:
                        counts[f"archive_{month}_rows"] += 1
                        day = row["FlightDate"][:10]
                        if not "2025-06-30" <= day <= "2025-08-01":
                            continue
                        counts["context_all_carriers"] += 1
                        if row["Dest"] == "JFK" and day.startswith("2025-07"):
                            arrival_counts[row["Reporting_Airline"]] += 1
                        if row["Reporting_Airline"] not in selected:
                            continue
                        counts["context_selected_carriers"] += 1
                        airports.update((row["Origin"], row["Dest"]))
                        writer.writerow({f: row[f] for f in fields})
            print(f"Verified and filtered {path.name}", flush=True)
    tz_path = reference / "openflights_airports.dat"
    zones = {}
    with tz_path.open(newline="") as handle:
        for row in csv.reader(handle):
            if len(row) > 11 and row[4] in airports and row[11] != "\\N":
                zones[row[4]] = row[11]
    missing = sorted(airports - zones.keys())
    # Missing zones are recorded, never guessed. The cleaner excludes affected tails.
    write_csv(reference / "timezones.csv", [dict(airport=a, timezone=z) for a, z in sorted(zones.items())])
    write_csv(reference / "terminals.csv", [dict(airline=a, terminal=t) for a, t in [("AA", "T8"), ("B6", "T5"), ("DL", "T4")]])
    manifest["timezone_reference"] = dict(url="https://openflights.org/data.php", sha256=sha256(tz_path),
                                           missing_airports=missing,
                                           license="Open Database License; see https://openflights.org/data.php#license")
    manifest["terminal_mapping"] = dict(mapping={"AA": "T8", "B6": "T5", "DL": "T4"},
        source="https://www.jfkairport.com/flights/airlines",
        caveat="Current published domestic-airline assignments used as a modeling mapping; not a historical gate-allocation record.")
    manifest["raw_counts"] = dict(counts)
    manifest["july_jfk_arrivals_by_reporting_carrier_before_screening"] = dict(arrival_counts)
    clocks = Counter()
    zone_objects = {a: ZoneInfo(z) for a, z in zones.items()}
    with rawcsv.open(newline="") as handle:
        for row in csv.DictReader(handle):
            if float(row["Cancelled"]) or float(row["Diverted"]):
                continue
            if row["Origin"] not in zone_objects or row["Dest"] not in zone_objects:
                clocks["missing_timezone"] += 1
                continue
            hhmm = int(row["CRSDepTime"])
            scheduled = datetime.fromisoformat(row["FlightDate"][:10]).replace(tzinfo=zone_objects[row["Origin"]])
            scheduled += timedelta(hours=hhmm // 100, minutes=hhmm % 100)
            departure = scheduled.astimezone(timezone.utc) + timedelta(minutes=float(row["DepDelay"]))
            arrival = (departure + timedelta(minutes=float(row["ActualElapsedTime"]))).astimezone(zone_objects[row["Dest"]])
            clock = int(row["ArrTime"])
            expected = (clock // 100 * 60 + clock % 100) % 1440
            delta = abs(arrival.hour * 60 + arrival.minute - expected)
            delta = min(delta, 1440 - delta)
            clocks["checked_completed_legs"] += 1
            clocks["maximum_clock_error_minutes"] = max(clocks["maximum_clock_error_minutes"], delta)
            if delta == 0:
                clocks["exact_arrival_clock_match"] += 1
            elif delta <= 1:
                clocks["one_minute_rounding_difference"] += 1
            else:
                clocks["arrival_clock_error_over_one_minute"] += 1
    manifest["arrival_clock_audit"] = dict(clocks)
    if clocks["arrival_clock_error_over_one_minute"]:
        raise ValueError("Arrival clocks inconsistent with timezone/elapsed reconstruction; inspect before modeling")
    audit = clean_bts(rawcsv, reference / "timezones.csv", reference / "terminals.csv",
                      "JFK", processed / "jfk_context_visits.csv")
    visits = [r for r in read_csv(processed / "jfk_context_visits.csv") if r["date"].startswith("2025-07")]
    if not visits:
        raise ValueError("No retained July visits; inspect cleaning audit")
    write_csv(processed / "jfk_2025_07.csv", visits)
    manifest["cleaning_audit"] = audit
    manifest["retained_july_visits"] = len(visits)
    manifest["retained_by_date"] = dict(sorted(Counter(r["date"] for r in visits).items()))
    manifest["retained_by_carrier"] = dict(Counter(r["airline"] for r in visits))
    manifest["cleaned_sha256"] = sha256(processed / "jfk_2025_07.csv")
    (processed / "bts_provenance.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
