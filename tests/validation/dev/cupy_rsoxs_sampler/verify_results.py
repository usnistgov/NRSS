"""Check registered CoreShell bytes or assess paired GPU timing summaries."""
import argparse
from collections import Counter
import json
import math
from pathlib import Path
from statistics import median


def read(path):
    return json.loads(path.read_text())


def core_bits(folder):
    import numpy as np

    manifests = [read(folder/f"r01-{arm}.result.json")["arrays"] for arm in "AB"]
    if manifests[0] != manifests[1]:
        raise ValueError("CoreShell result names, dimensions, or shapes changed")
    counts = {}
    with np.load(folder/"r01-A.npz", allow_pickle=False) as a, np.load(folder/"r01-B.npz", allow_pickle=False) as b:
        if set(a.files) != set(b.files):
            raise ValueError("Saved array names changed")
        for group in ("coords", "values"):
            for name, info in manifests[0][group].items():
                key = info["key"]
                left, right = a[key], b[key]
                if left.dtype != right.dtype or left.shape != right.shape:
                    raise ValueError(f"{key}: dtype or shape changed")
                if group == "values" and "energy" in info["dims"]:
                    axis = info["dims"].index("energy")
                    panels = zip(np.moveaxis(left, axis, 0), np.moveaxis(right, axis, 0))
                else:
                    panels = [(left, right)]
                counts[key] = []
                for i, (x, y) in enumerate(panels):
                    # Compare integer representations, including NaN payload and signed zero.
                    xx = np.ascontiguousarray(x).view(np.uint8).reshape(x.size, x.dtype.itemsize)
                    yy = np.ascontiguousarray(y).view(np.uint8).reshape(y.size, y.dtype.itemsize)
                    bad = np.flatnonzero(np.any(xx != yy, axis=1))
                    if bad.size:
                        idx = int(bad[0])
                        raise ValueError(f"{key}, panel {i}, flat index {idx}: {len(bad)} mismatches, "
                                         f"{xx[idx].tobytes().hex()} vs {yy[idx].tobytes().hex()}")
                    counts[key].append({"samples": x.size, "mismatches": 0})
    return counts


def topology(folder):
    reports = [read(folder/f"r01-{k}.result.json")["backend_timing_details"]["execution"] for k in "AB"]
    if reports[0] != reports[1]:
        raise ValueError(f"CoreShell execution topology changed in {folder}")
    data = reports[0]
    if data["routes"] != ["compact_r2c"] or data["basis_builds"] != 1:
        raise ValueError(f"Unexpected CoreShell route or basis build count: {data}")
    return data


def core_memory(folder):
    peaks = {}
    for arm in "AB":
        worker = read(folder/f"r01-{arm}.result.json")
        gpu = worker["memory"].get("peak_gpu_worker_mib")
        if gpu is None:
            raise ValueError(f"Missing per-PID NVML peak for {arm} in {folder}")
        peaks[arm] = dict(pid=worker["worker_pid"], peak_gpu_mib=gpu,
                          peak_host_mib=worker["peak_host_rss_mib"],
                          sample_count=worker["memory"]["sample_count"],
                          poll_seconds=worker["memory"]["sample_interval_s"])
    return peaks


def ratios(paths, metric):
    values = []
    for path in paths:
        summary_path = path / "summary.json" if path.is_dir() else path
        summary = read(summary_path)
        if metric == "first_seconds" and metric not in summary:
            # The native driver retains first_call_seconds in the individual worker JSONs.
            if summary.get("mode") != "timing":
                raise ValueError(f"Expected an uninstrumented timing run at {path}")
            rows = [{arm: read(summary_path.parent/f"r{i:02d}-{arm}.result.json")["first_call_seconds"]
                     for arm in "AB"} for i in range(1, 1+len(summary["primary_seconds"]))]
        else:
            rows = summary[metric]
        for row in rows:
            if any(not math.isfinite(row[i]) or row[i] <= 0 for i in "AB"):
                raise ValueError(f"Missing or nonpositive paired timing in {path}")
            values.append(row["B"] / row["A"])
    return values


def interval(values):
    # Nonparametric central median interval; each excluded binomial tail <= 5%.
    ordered = sorted(values)
    n = len(ordered)
    tail, k = 0, 0
    while k < n//2 and (tail + math.comb(n, k))*20 <= 2**n:
        tail += math.comb(n, k)
        k += 1
    if not k:
        raise ValueError("At least five independent worker pairs are needed for this median interval")
    return ordered[k-1], ordered[-k], 1 - 2*tail/(2**n)


def assess(aa, ab, margin):
    lo, hi = 1-margin/100, 1+margin/100
    cal_l, cal_u, cal_coverage = interval(aa)
    lb, ub, coverage = interval(ab)
    if cal_l < lo or cal_u > hi or not cal_l <= 1 <= cal_u:
        decision = "inconclusive-baseline-noise-or-order-bias"
    elif ub < 1:
        decision = "resolved-faster-review"
    elif lb > 1:
        decision = "resolved-slower-hold"
    elif lb >= lo and ub <= hi:
        decision = "neutral-within-margin"
    else:
        decision = "inconclusive-more-pairs"
    return dict(decision=decision, margin_percent=margin, ratio_definition="candidate_seconds / parent_seconds",
                aa_pairs=len(aa), aa_median=median(aa), aa_interval=[cal_l, cal_u], aa_coverage=cal_coverage,
                ab_pairs=len(ab), ab_median=median(ab), percent_change=100*(median(ab)-1),
                ab_interval=[lb, ub], ab_coverage=coverage,
                pair_counts=dict(Counter("slower" if r>1 else "faster" if r<1 else "same" for r in ab)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    subs = parser.add_subparsers(dest="action", required=True)
    subs.add_parser("core-bits").add_argument("folder", type=Path)
    subs.add_parser("core-stages").add_argument("folder", type=Path)
    timing = subs.add_parser("timing")
    timing.add_argument("--aa", type=Path, nargs="+", required=True, help="A/A run folders or summaries")
    timing.add_argument("--ab", type=Path, nargs="+", required=True, help="Matched A/B run folders or summaries")
    timing.add_argument("--margin-percent", type=float, default=2)
    timing.add_argument("--metric", choices=("primary_seconds", "first_seconds"), default="primary_seconds")
    args = parser.parse_args()
    if args.action == "core-bits":
        report = core_bits(args.folder)
    elif args.action == "core-stages":
        report = topology(args.folder)
    else:
        if not 0 < args.margin_percent < 100:
            parser.error("Choose the percentage margin in advance, between zero and 100")
        report = assess(ratios(args.aa, args.metric), ratios(args.ab, args.metric), args.margin_percent)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
