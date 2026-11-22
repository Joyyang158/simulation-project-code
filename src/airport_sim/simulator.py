"""Discrete-event engine. Assignment never inspects future service durations."""
from dataclasses import dataclass, field
import heapq
import math
import numpy as np


@dataclass(frozen=True)
class Flight:
    id: str
    arrival: float
    service: float
    terminal: str = "T1"
    airline: str = "A"


@dataclass(frozen=True)
class Gate:
    id: str
    terminal: str
    airlines: tuple[str, ...] = ()  # Empty means all airlines compatible.
    closures: tuple[tuple[float, float], ...] = ()
    initial_busy_until: float = 0.0

    def compatible(self, flight):
        return not self.airlines or flight.airline in self.airlines

    def can_assign(self, time, transfer):
        # Do not reserve a gate across a closure during transfer. Existing
        # service is nonpreemptive and may finish after a closure begins.
        return not any(start <= time < end or time < start <= time + transfer
                       for start, end in self.closures)


@dataclass
class Result:
    flights: list[dict]
    metrics: dict
    trace: list[dict] = field(default_factory=list)
    intervals: list[dict] = field(default_factory=list)


def simulate(flights, gates, policy="pooled", horizon=1440.0,
             transfer_time=0.0, trace=False):
    """Arrivals in [0,horizon); drain to completion after observation ends.

    pooled: terminal FCFS; dedicated: round-robin fixed home gate on arrival;
    sharing: terminal FCFS with compatible free overflow gates.
    Scan waiting flights in arrival order, allowing an incompatible head to be
    bypassed. A transfer reserves its target immediately. Gate occupancy includes
    the caller's clearance buffer. Queue length excludes aircraft in transfer.
    """
    if policy not in {"pooled", "dedicated", "sharing"}:
        raise ValueError("Unknown gate policy")
    if not math.isfinite(horizon) or horizon <= 0 or not math.isfinite(transfer_time) or transfer_time < 0:
        raise ValueError("Invalid horizon or transfer time")
    flights = sorted(flights, key=lambda f: (f.arrival, f.id))
    gates = sorted(gates, key=lambda g: g.id)
    if not gates or len({g.id for g in gates}) != len(gates):
        raise ValueError("Gate IDs must be unique and at least one gate is required")
    if len({f.id for f in flights}) != len(flights):
        raise ValueError("Flight IDs must be unique")
    for g in gates:
        if not math.isfinite(g.initial_busy_until) or g.initial_busy_until < 0:
            raise ValueError("Invalid initial occupancy")
        last_end = -1.0
        for start, end in sorted(g.closures):
            if not all(map(math.isfinite, (start, end))) or start < 0 or start >= end or start < last_end:
                raise ValueError("Closures must be finite, nonnegative, and nonoverlapping")
            last_end = end
    for f in flights:
        if not (math.isfinite(f.arrival) and 0 <= f.arrival < horizon and
                math.isfinite(f.service) and f.service > 0):
            raise ValueError("Flights require arrivals in [0,horizon) and positive finite service")
        if not any(g.terminal == f.terminal and g.compatible(f) for g in gates):
            raise ValueError(f"No compatible home gate for {f.id}")

    events, sequence = [], 0
    def schedule(time, kind, payload):
        nonlocal sequence
        heapq.heappush(events, (time, sequence, kind, payload))
        sequence += 1

    busy = {g.id: False for g in gates}
    by_id = {g.id: g for g in gates}
    intervals, logs, records, waiting = [], [], {}, []
    assigned, counters = {}, {}
    for f in flights:
        schedule(f.arrival, "arrival", f)
        if policy == "dedicated":
            choices = [g.id for g in gates if g.terminal == f.terminal and g.compatible(f)]
            key = tuple(choices)
            index = counters.get(key, 0)
            assigned[f.id] = choices[index % len(choices)]
            counters[key] = index + 1
    for g in gates:
        if g.initial_busy_until:
            busy[g.id] = True
            intervals.append(dict(gate=g.id, flight="INITIAL", reserved=0.0,
                                  start=0.0, end=g.initial_busy_until))
            schedule(g.initial_busy_until, "release", g.id)
        for start, end in g.closures:
            schedule(start, "closure_start", g.id)
            schedule(end, "closure_end", g.id)
    schedule(horizon, "observation_end", None)
    previous, queue_area, max_queue = 0.0, 0.0, 0
    while events:
        time = events[0][0]
        queue_area += len(waiting) * max(0.0, min(time, horizon) - min(previous, horizon))
        previous = time
        # Batch all simultaneous events before allocating gates.
        while events and events[0][0] == time:
            _, _, kind, payload = heapq.heappop(events)
            if kind == "arrival":
                waiting.append(payload)
            elif kind == "release":
                busy[payload] = False
            if trace:
                logs.append(dict(time=time, event=kind,
                                 subject=payload.id if isinstance(payload, Flight) else payload))
        waiting.sort(key=lambda f: (f.arrival, f.id))
        remaining = []
        for f in waiting:
            eligible = [g for g in gates if not busy[g.id] and g.compatible(f)]
            home = [g for g in eligible if g.terminal == f.terminal and g.can_assign(time, 0)]
            if policy == "dedicated":
                home = [g for g in home if g.id == assigned[f.id]]
            options, travel = home, 0.0
            if not options and policy == "sharing":
                options = [g for g in eligible if g.terminal != f.terminal and g.can_assign(time, transfer_time)]
                travel = transfer_time
            if not options:
                remaining.append(f)
                continue
            g = options[0]  # Deterministic ID tie-break; no look-ahead.
            busy[g.id] = True
            start = time + travel
            end = start + f.service
            row = dict(id=f.id, terminal=f.terminal, airline=f.airline,
                       gate=g.id, gate_terminal=g.terminal, arrival=f.arrival,
                       assignment=time, start=start, end=end, service=f.service,
                       queue_wait=time - f.arrival, transfer=travel,
                       wait=start - f.arrival, overflow=int(g.terminal != f.terminal))
            records[f.id] = row
            intervals.append(dict(gate=g.id, flight=f.id, reserved=time, start=start, end=end))
            if travel:
                schedule(start, "service_start", f.id)
            schedule(end, "release", g.id)
            if trace:
                logs.append(dict(time=time, event="assign", subject=f.id, gate=g.id,
                                 service_start=start, release=end))
        waiting = remaining
        if time < horizon:
            max_queue = max(max_queue, len(waiting))
    if waiting:
        raise RuntimeError("Unserved flights remain after the final event")
    rows = [records[f.id] for f in flights]
    waits = np.array([r["wait"] for r in rows])
    def mean(key):
        return float(np.mean([r[key] for r in rows])) if rows else 0.0
    def area(key):
        return sum(max(0.0, min(i["end"], horizon) - min(i[key], horizon)) for i in intervals)
    metrics = dict(n_flights=len(rows), mean_wait=mean("wait"),
                   p95_wait=float(np.quantile(waits, .95)) if rows else 0.0,
                   prob_wait_gt15=float(np.mean(waits > 15)) if rows else 0.0,
                   mean_queue_wait=mean("queue_wait"), mean_transfer=mean("transfer"),
                   overflow_share=mean("overflow"),
                   mean_queue=queue_area / horizon, max_queue=max_queue,
                   utilization=area("start") / (len(gates) * horizon),
                   reserved_utilization=area("reserved") / (len(gates) * horizon),
                   completed_in_day=sum(r["end"] <= horizon for r in rows),
                   completed_after_drain=len(rows),
                   last_completion=max((r["end"] for r in rows), default=0.0))
    return Result(rows, metrics, logs, intervals)
