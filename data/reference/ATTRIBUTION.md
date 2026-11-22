# Input reference provenance

The flight records are the U.S. Bureau of Transportation Statistics Reporting
Carrier On-Time Performance archives for June, July, and August 2025:
https://www.transtats.bts.gov/PREZIP/

Download date: 2026-11-22 UTC (2026-11-21 America/New_York).
Archive URLs, sizes, SHA-256 checksums, row counts, and cleaning audit are recorded
in `data/processed/bts_provenance.json`.

`openflights_airports.dat` is an OpenFlights airport database snapshot downloaded
from https://raw.githubusercontent.com/jpatokal/openflights/master/data/airports.dat.
Documentation and attribution: https://openflights.org/data.php.
OpenFlights makes its database available under the Open Database License (ODbL):
https://opendatacommons.org/licenses/odbl/1-0/.
Individual database contents are under the Database Contents License:
https://opendatacommons.org/licenses/dbcl/1-0/.
The derived `timezones.csv` mapping is provided under those same database terms.
The script uses the named IANA time zones with date-specific ZoneInfo offsets,
not OpenFlights' static UTC-offset/DST columns.

`terminals.csv` uses AA/T8, B6/T5, and DL/T4 as modeling groups, consistent with
the current published JFK airline list:
https://www.jfkairport.com/flights/airlines.
This is not a verified historical July 2025 gate assignment or a gate-capacity
inventory. Shared-gate eligibility and four gates per group are hypothetical.
