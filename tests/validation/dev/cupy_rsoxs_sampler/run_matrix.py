"""Launch the specified qualification matrix with the repository's GPU workers."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

from run_lipid_pair import CASES as LIPID_CASES
from verify_results import assess, core_bits, core_memory, ratios, topology

HERE = Path(__file__).resolve().parent
NATIVE = ("core-small", "core-medium", "core-large")
PRIMARY = (*(name + "-one" for name in NATIVE), *NATIVE, *LIPID_CASES)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--parent-root", type=Path, required=True)
    p.add_argument("--candidate-root", type=Path, required=True)
    p.add_argument("--manifest", type=Path, help="Frozen ZP09 fixture.json for lipid rows")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--phase", choices=("plan", "check", "stages", "time", "memory"), required=True)
    p.add_argument("--pairing", nargs="+", choices=("aa", "ab"), default=("aa", "ab"),
                   help="Run the baseline controls and candidate pairs together or in separate batches")
    p.add_argument("--case", nargs="+", choices=PRIMARY, default=PRIMARY)
    p.add_argument("--pairs", type=int, default=8)
    p.add_argument("--warms", type=int, default=5, help="After-first runs per frozen lipid worker")
    p.add_argument("--margin-percent", type=float, default=2.0)
    p.add_argument("--gpu", default=os.environ.get("CUDA_VISIBLE_DEVICES") or "0")
    args = p.parse_args()
    if args.phase == "time" and args.pairs < 5:
        p.error("Use at least five independent pairs for a timing decision")
    if not (0 < args.margin_percent < 100) or args.warms < 1 or args.pairs < 1:
        p.error("Invalid timing settings")
    if not args.gpu.strip() or "," in args.gpu:
        p.error("Pin one physical GPU")
    parent, candidate = args.parent_root.resolve(), args.candidate_root.resolve()
    if parent == candidate and "ab" in args.pairing and args.phase != "plan":
        p.error("Use different worktrees for the two revisions")
    if any(case in LIPID_CASES for case in args.case):
        if not args.manifest or (args.phase != "plan" and not args.manifest.is_file()):
            p.error("Supply the frozen fixture.json for lipid; --case can select native rows independently")
    args.out.mkdir(parents=True, exist_ok=True)
    folder = Path(tempfile.mkdtemp(prefix=args.phase+"-", dir=args.out.resolve()))
    print(f"Evidence: {folder}", flush=True)
    native = parent/"tests/validation/dev/cupy_rsoxs_optimization/run_speed_tickets.py"
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": args.gpu, "PYTHONDONTWRITEBYTECODE": "1"}
    decisions = {}
    for case in args.case:
        dirs = {}
        for pairing in dict.fromkeys(args.pairing):
            output = folder/case/pairing
            output.mkdir(parents=True)
            if case.startswith("core-"):
                size = case.split("-")[1]
                one = case.endswith("-one")
                cmd = [sys.executable, str(native), "--case", "one" if one else "energy101",
                       "--rotation", "off" if one else "12", "--fixture", "core_shell",
                       "--shape-label", size, "--path", "default", "--resident", "host",
                       "--result-residency", "host", "--layout", "detector", "--result-chunk-size", "1",
                       "--baseline-root", str(parent), "--out-root", str(output), "--gpu", args.gpu,
                       "--repeats", str(args.pairs if args.phase in ("time", "plan") else 1)]
                cmd += {"plan": ["--dry-run"], "time": [], "check": ["--accuracy", "full"],
                        "stages": ["--stages", "--expect-route", "compact_r2c"],
                        "memory": ["--memory", "--memory-poll-interval", "0.005"]}[args.phase]
            else:
                cmd = [sys.executable, str(HERE/"run_lipid_pair.py"), "--case", case,
                       "--manifest", str(args.manifest.resolve()), "--parent-root", str(parent),
                       "--out", str(output), "--phase", args.phase, "--pairs", str(args.pairs),
                       "--warms", str(args.warms), "--gpu", args.gpu]
            cmd += ["--aa"] if pairing == "aa" else ["--candidate-root", str(candidate)]
            (output/"command.json").write_text(json.dumps(cmd, indent=2)+"\n")
            print(f"{case} {pairing}", flush=True)
            with (output/"launcher.log").open("w") as log:
                subprocess.run(cmd, cwd=parent, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
            if args.phase == "plan":
                continue
            runs = [x for x in output.iterdir() if x.is_dir()]
            if len(runs) != 1:
                raise RuntimeError(f"Expected one new worker run under {output}")
            dirs[pairing] = runs[0]
            core_checks = {"check": core_bits, "stages": topology, "memory": core_memory}
            if case.startswith("core-") and args.phase in core_checks:
                result = core_checks[args.phase](runs[0])
                (output/"verification.json").write_text(json.dumps(result, indent=2)+"\n")
        if args.phase == "time" and set(dirs) == {"aa", "ab"}:
            decisions[case] = {name: assess(ratios([dirs["aa"]], name), ratios([dirs["ab"]], name),
                                            args.margin_percent) for name in ("primary_seconds", "first_seconds")}
            (folder/"timing-decisions.json").write_text(json.dumps(decisions, indent=2)+"\n")
            for metric, result in decisions[case].items():
                print(f"{case} {metric}: {result['decision']}, {result['percent_change']:+.2f}%", flush=True)
    if any(row["decision"] != "neutral-within-margin" for case in decisions.values() for row in case.values()):
        raise SystemExit(2)


if __name__ == "__main__":
    main()
