import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
REPORT_DIR = REPO_ROOT / "test-reports" / "2d-validation-replicates-dev"
JSON_PREFIX = "NRSS_JSON:"
PAIRINGS = [
    ("delta_pos_vac", "delta_neg_vac"),
    ("delta_pos_split", "delta_neg_split"),
    ("beta_vac", "beta_split"),
    ("delta_pos_vac", "delta_pos_split"),
    ("delta_neg_vac", "delta_neg_split"),
    ("mixed_vac", "mixed_split"),
]


def _child_env() -> dict[str, str]:
    env = os.environ.copy()
    env.setdefault("CUDA_VISIBLE_DEVICES", "0")
    return env


def _run_child(code: str) -> tuple[dict[str, object], str]:
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=str(REPO_ROOT),
        env=_child_env(),
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    output = result.stdout or ""
    payload = None
    for line in output.splitlines():
        if line.startswith(JSON_PREFIX):
            payload = json.loads(line[len(JSON_PREFIX) :])
    if result.returncode != 0:
        raise RuntimeError(
            f"Child run failed with exit code {result.returncode}.\nOutput:\n{output}"
        )
    if payload is None:
        raise RuntimeError(f"Child run did not emit {JSON_PREFIX!r} payload.\nOutput:\n{output}")
    return payload, output


def _geometry_child_code(diameter_nm: float) -> str:
    return f"""
import json
from tests.validation.test_analytical_2d_disk_form_factor import _evaluate_geometry_case

result = _evaluate_geometry_case({float(diameter_nm)!r})
payload = {{
    "diameter_nm": {float(diameter_nm)!r},
    "rms_log": float(result["assert_metrics"]["rms_log"]),
    "p95_log_abs": float(result["assert_metrics"]["p95_log_abs"]),
    "max_log_abs": float(result["assert_metrics"]["max_log_abs"]),
    "min_mae_abs_dq": float(result["assert_minima_metrics"]["mae_abs_dq"]),
    "min_rmse_abs_dq": float(result["assert_minima_metrics"]["rmse_abs_dq"]),
    "min_max_abs_dq": float(result["assert_minima_metrics"]["max_abs_dq"]),
    "n_match": int(result["assert_minima_metrics"]["n_match"]),
    "sim_seconds": float(result["timing_by_superres"][1]["sim_seconds"]),
    "iq_seconds": float(result["timing_by_superres"][1]["iq_seconds"]),
}}
print("{JSON_PREFIX}" + json.dumps(payload, sort_keys=True))
"""


def _contrast_child_code() -> str:
    return f"""
import json
from tests.validation.test_2d_disk_contrast_scaling import _evaluate_contrast_family_rows

family_rows = _evaluate_contrast_family_rows()
pairings = {PAIRINGS!r}

max_weighted_rel_err = 0.0
max_unweighted_rel_err = 0.0
max_integral_consistency_rel = 0.0
for rows in family_rows.values():
    for row in rows:
        max_weighted_rel_err = max(max_weighted_rel_err, float(row["weighted_rel_err"]))
        max_unweighted_rel_err = max(max_unweighted_rel_err, float(row["unweighted_rel_err"]))
        max_integral_consistency_rel = max(
            max_integral_consistency_rel,
            float(row["integral_consistency_rel"]),
        )

max_pairing_rel_err = 0.0
for left_family, right_family in pairings:
    left_rows = family_rows[left_family]
    right_rows = family_rows[right_family]
    for left_row, right_row in zip(left_rows, right_rows):
        rel = abs(left_row["weighted_ratio"] - right_row["weighted_ratio"]) / left_row["weighted_ratio"]
        max_pairing_rel_err = max(max_pairing_rel_err, float(rel))

payload = {{
    "max_weighted_rel_err": max_weighted_rel_err,
    "max_unweighted_rel_err": max_unweighted_rel_err,
    "max_integral_consistency_rel": max_integral_consistency_rel,
    "max_pairing_rel_err": max_pairing_rel_err,
}}
print("{JSON_PREFIX}" + json.dumps(payload, sort_keys=True))
"""


def _summarize_records(records: list[dict[str, object]], metric_names: list[str]) -> dict[str, dict[str, float]]:
    summary = {}
    for metric_name in metric_names:
        values = np.asarray([float(record[metric_name]) for record in records], dtype=np.float64)
        summary[metric_name] = {
            "min": float(np.min(values)),
            "mean": float(np.mean(values)),
            "max": float(np.max(values)),
            "std": float(np.std(values)),
        }
    return summary


def _write_geometry_plot(geometry_records: dict[str, list[dict[str, object]]]) -> Path:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    metrics = [
        ("rms_log", "RMS log error"),
        ("p95_log_abs", "95th pct abs log error"),
        ("min_mae_abs_dq", "Minima MAE |dq|"),
        ("min_rmse_abs_dq", "Minima RMSE |dq|"),
    ]
    diameters = sorted(geometry_records, key=float)

    fig, axes = plt.subplots(2, 2, figsize=(11.5, 8.5))
    for ax, (metric_name, title) in zip(axes.flat, metrics):
        series = [
            np.asarray([float(record[metric_name]) for record in geometry_records[diameter]], dtype=np.float64)
            for diameter in diameters
        ]
        ax.boxplot(series, tick_labels=[f"{float(d):.0f} nm" for d in diameters])
        for idx, values in enumerate(series, start=1):
            jitter = np.linspace(-0.06, 0.06, values.size) if values.size > 1 else np.asarray([0.0])
            ax.scatter(np.full(values.shape, idx, dtype=np.float64) + jitter, values, s=18, alpha=0.75)
        ax.set_title(title)
        ax.grid(alpha=0.25)

    fig.suptitle("2D Analytical Disk Replicate Metrics")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    out = REPORT_DIR / "geometry_replicate_distributions.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def _write_contrast_plot(contrast_records: list[dict[str, object]]) -> Path:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    if not contrast_records:
        raise AssertionError("No contrast records available for plotting.")
    metrics = [
        ("max_weighted_rel_err", "Max weighted ratio rel err"),
        ("max_unweighted_rel_err", "Max unweighted ratio rel err"),
        ("max_integral_consistency_rel", "Max weighted/unweighted consistency err"),
        ("max_pairing_rel_err", "Max family pairing rel err"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 8.5))
    x = np.arange(1, len(contrast_records) + 1, dtype=np.int32)

    for ax, (metric_name, title) in zip(axes.flat, metrics):
        y = np.asarray([float(record[metric_name]) for record in contrast_records], dtype=np.float64)
        ax.plot(x, y, marker="o", linewidth=1.1)
        ax.set_title(title)
        ax.set_xlabel("Replicate")
        ax.grid(alpha=0.25)

    fig.suptitle("2D Contrast-Scaling Replicate Metrics")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    out = REPORT_DIR / "contrast_replicate_traces.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run replicate-calibration campaigns for the 2D NRSS validation tests."
    )
    parser.add_argument("--geometry-reps", type=int, default=10)
    parser.add_argument("--contrast-reps", type=int, default=10)
    args = parser.parse_args()

    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    geometry_records: dict[str, list[dict[str, object]]] = {}
    for diameter_nm in (70.0, 128.0):
        key = f"{diameter_nm:.1f}"
        geometry_records[key] = []
        for replicate in range(1, args.geometry_reps + 1):
            payload, _ = _run_child(_geometry_child_code(diameter_nm))
            payload["replicate"] = replicate
            geometry_records[key].append(payload)
            print(
                "geometry "
                f"d={diameter_nm:.0f} rep={replicate}/{args.geometry_reps} "
                f"rms={payload['rms_log']:.4f} p95={payload['p95_log_abs']:.4f} "
                f"min_mae={payload['min_mae_abs_dq']:.5f} min_rmse={payload['min_rmse_abs_dq']:.5f}"
            )

    contrast_records: list[dict[str, object]] = []
    contrast_failure = None
    for replicate in range(1, args.contrast_reps + 1):
        try:
            payload, _ = _run_child(_contrast_child_code())
        except RuntimeError as exc:
            contrast_failure = {"replicate": replicate, "error": str(exc)}
            print(f"contrast rep={replicate}/{args.contrast_reps} FAILED")
            break
        payload["replicate"] = replicate
        contrast_records.append(payload)
        print(
            "contrast "
            f"rep={replicate}/{args.contrast_reps} "
            f"werr={payload['max_weighted_rel_err']:.5f} "
            f"uerr={payload['max_unweighted_rel_err']:.5f} "
            f"cerr={payload['max_integral_consistency_rel']:.5f} "
            f"perr={payload['max_pairing_rel_err']:.5f}"
        )

    geometry_summary = {
        diameter: _summarize_records(
            records,
            ["rms_log", "p95_log_abs", "min_mae_abs_dq", "min_rmse_abs_dq", "sim_seconds", "iq_seconds"],
        )
        for diameter, records in geometry_records.items()
    }
    contrast_summary = _summarize_records(
        contrast_records,
        [
            "max_weighted_rel_err",
            "max_unweighted_rel_err",
            "max_integral_consistency_rel",
            "max_pairing_rel_err",
        ],
    ) if contrast_records else {}

    _write_geometry_plot(geometry_records)
    if contrast_records:
        _write_contrast_plot(contrast_records)

    with open(REPORT_DIR / "geometry_replicates.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "geometry_reps": args.geometry_reps,
                "records": geometry_records,
                "summary": geometry_summary,
            },
            f,
            indent=2,
            sort_keys=True,
        )
    with open(REPORT_DIR / "contrast_replicates.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "contrast_reps": args.contrast_reps,
                "records": contrast_records,
                "summary": contrast_summary,
                "failure": contrast_failure,
            },
            f,
            indent=2,
            sort_keys=True,
        )

    print("geometry_summary")
    print(json.dumps(geometry_summary, indent=2, sort_keys=True))
    print("contrast_summary")
    print(json.dumps(contrast_summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
