"""Join completed detector A/B checks, timing and memory reports into a CSV."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import statistics
import sys


def median(values):
    return statistics.median(values) if values else ""


def collect(root):
    groups = {}
    for path in sorted(root.rglob("plan.json")):
        plan = json.loads(path.read_text())
        summary = path.with_name("summary.json")
        if not summary.is_file() or "cases" not in plan:
            continue
        results = json.loads(summary.read_text())
        folder, phase = path.parent, plan["phase"]
        for case in plan["cases"]:
            route = case["route"]
            if route not in results or phase not in {"time", "memory", "check", "diagnose"}:
                continue
            sample = {a: json.loads((folder / f"{route}-00-{a}.json").read_text()) for a in ("A", "B")}
            env = [(sample[a]["sha"], sample[a]["gpu"], sample[a]["cupy"], sample[a]["driver"],
                    sample[a]["runtime"], sample[a]["host"], sample[a]["source_sha256"]) for a in ("A", "B")]
            workload = {key: value for key, value in case.items() if key != "warms"}
            signature = json.dumps([workload, env, plan["aa"], plan["gpu"]], sort_keys=True)
            group = groups.setdefault(signature, {"case": case, "sample": sample, "aa": plan["aa"],
                                                   "times": [], "memory": [], "checked": [], "folders": []})
            group["folders"].append(str(folder))
            if phase == "check":
                group["checked"].append(sum(c["in_window_values_checked"] for c in results[route].values()))
            if phase == "memory":
                group["memory"].append(sample)
            if phase == "time":
                for a_path in sorted(folder.glob(f"{route}-*-A.json")):
                    b_path = a_path.with_name(a_path.name[:-6] + "B.json")
                    group["times"].append((json.loads(a_path.read_text()), json.loads(b_path.read_text())))
    for group in groups.values():
        case, sample, times = group["case"], group["sample"], group["times"]
        row = {"route": case["route"], "source_zyx": "x".join(map(str, case["source"])),
               "logical_zyx": "x".join(map(str, case["target"])), "q_max_nm_inv": case["q_max"],
               "energies": len(case["energies"]), "angle_start_inc_end": str(case["rotation"]),
               "residency": case["residency"], "layout": case["layout"], "aa": group["aa"],
               "grid": case["grid"], "chunk": case["chunk"], "fill": case["fill"], "window": case["window"],
               "rot_mask": case["rot_mask"], "gpu": sample["A"]["gpu"],
               "energy_list_hash": hashlib.sha256(json.dumps(case["energies"]).encode()).hexdigest()[:16]}
        for arm, index in (("A", 0), ("B", 1)):
            row[arm + "_sha"] = sample[arm]["sha"]
            row[arm + "_source_sha256"] = sample[arm]["source_sha256"]
            row[arm + "_first_ms"] = median([1e3*t[index]["first_run_s"] for t in times])
            row[arm + "_warm_ms"] = median([1e3*median(t[index]["warm_run_s"]) for t in times])
            row[arm + "_output_bytes"] = sample[arm]["return_bytes"]
            row[arm + "_output_MiB"] = sample[arm]["return_bytes"] / 2**20
        ratios = [median(a["warm_run_s"]) / median(b["warm_run_s"]) for a, b in times]
        row["warm_paired_speedup"] = median(ratios)
        row["warm_paired_speedup_min"] = min(ratios, default="")
        row["warm_paired_speedup_max"] = max(ratios, default="")
        for kind in ("host", "gpu"):
            key = f"peak_{kind}_mib"
            for arm in ("A", "B"):
                row[f"{arm}_{key}"] = median([r[arm][key] for r in group["memory"]])
            old, new = row[f"A_{key}"], row[f"B_{key}"]
            row[f"{kind}_saved_MiB"] = old-new if old != "" else ""
            row[f"{kind}_saved_percent"] = 100*(old-new)/old if old else ""
        row["in_window_values_verified"] = max(group["checked"], default="")
        row["bit_mismatches"] = 0 if group["checked"] else ""
        row["worker_directories"] = " | ".join(group["folders"])
        yield row


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    args = parser.parse_args()
    rows = list(collect(args.root))
    if not rows:
        parser.exit(1, "No completed comparison reports found.\n")
    writer = csv.DictWriter(sys.stdout, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
