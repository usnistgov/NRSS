#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt


REPO_ROOT = Path(__file__).resolve().parents[4]
SRC_PATH = REPO_ROOT / "src"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")

from tests.validation.dev.cupy_rsoxs_optimization.run_cupy_rsoxs_optimization_matrix import (  # noqa: E402
    SIZE_SPECS,
    _json_default,
    _timestamp,
)
from tests.validation.lib.core_shell import has_visible_gpu  # noqa: E402


OUT_ROOT = REPO_ROOT / "test-reports" / "core-shell-backend-performance-dev"
SUMMARY_NAME = "primary_backend_speed_comparison_summary.json"
TABLE_NAME = "primary_backend_speed_comparison_table.tsv"
FIGURE_NAME = "primary_backend_speed_comparison_table.png"
LEGACY_ROOT = REPO_ROOT / "test-reports" / "cyrsoxs-timing-dev"
CUPY_ROOT = REPO_ROOT / "test-reports" / "cupy-rsoxs-optimization-dev"
LEGACY_SCRIPT = (
    REPO_ROOT / "tests" / "validation" / "dev" / "cyrsoxs_timing" / "run_cyrsoxs_timing_matrix.py"
)
CUPY_SCRIPT = (
    REPO_ROOT
    / "tests"
    / "validation"
    / "dev"
    / "cupy_rsoxs_optimization"
    / "run_cupy_rsoxs_optimization_matrix.py"
)

LANE_ORDER = ("small", "medium", "large")
STARTUP_ORDER = ("cold", "pre-warm")
ENERGY_ORDER = ("single", "triple")
ROTATION_ORDER = (
    (
        "no rotation",
        {"single": "single_no_rotation", "triple": "triple_no_rotation"},
        [0.0, 0.0, 0.0],
    ),
    (
        "some rotation (0,15,165)",
        {"single": "single_rot_0_15_165", "triple": "triple_limited_rotation"},
        [0.0, 15.0, 165.0],
    ),
)


@dataclass(frozen=True)
class ComponentSpec:
    key: str
    run_label_suffix: str
    script_path: Path
    output_root: Path
    args: tuple[str, ...]


COMPONENT_SPECS = (
    ComponentSpec(
        key="legacy_cold",
        run_label_suffix="legacy_cold",
        script_path=LEGACY_SCRIPT,
        output_root=LEGACY_ROOT,
        args=(
            "--size-labels",
            "small,medium,large",
            "--rotation-specs",
            "0:15:165",
            "--include-triple-no-rotation",
            "--include-triple-limited",
        ),
    ),
    ComponentSpec(
        key="legacy_prewarm",
        run_label_suffix="legacy_prewarm",
        script_path=LEGACY_SCRIPT,
        output_root=LEGACY_ROOT,
        args=(
            "--size-labels",
            "small,medium,large",
            "--rotation-specs",
            "0:15:165",
            "--include-triple-no-rotation",
            "--include-triple-limited",
            "--cuda-prewarm",
            "before_prepare_inputs",
        ),
    ),
    ComponentSpec(
        key="cupy_host_and_device_cold",
        run_label_suffix="cupy_host_and_device_cold",
        script_path=CUPY_SCRIPT,
        output_root=CUPY_ROOT,
        args=(
            "--resident-modes",
            "host,device",
            "--size-labels",
            "small,medium,large",
            "--rotation-specs",
            "0:15:165",
            "--include-triple-no-rotation",
            "--include-triple-limited",
        ),
    ),
    ComponentSpec(
        key="cupy_host_prewarm",
        run_label_suffix="cupy_host_prewarm",
        script_path=CUPY_SCRIPT,
        output_root=CUPY_ROOT,
        args=(
            "--resident-modes",
            "host",
            "--size-labels",
            "small,medium,large",
            "--rotation-specs",
            "0:15:165",
            "--include-triple-no-rotation",
            "--include-triple-limited",
            "--cuda-prewarm",
            "before_prepare_inputs",
        ),
    ),
)


def _summary_path(output_root: Path, run_label: str) -> Path:
    return output_root / run_label / "summary.json"


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def _component_run_label(label: str, spec: ComponentSpec) -> str:
    return f"{label}__{spec.run_label_suffix}"


def _run_component(spec: ComponentSpec, *, label: str, force_rerun: bool) -> dict[str, Any]:
    run_label = _component_run_label(label, spec)
    summary_path = _summary_path(spec.output_root, run_label)
    if summary_path.exists() and not force_rerun:
        return {
            "key": spec.key,
            "run_label": run_label,
            "summary_path": str(summary_path),
            "reused_existing": True,
            "summary": _load_json(summary_path),
        }

    cmd = [
        sys.executable,
        str(spec.script_path),
        "--label",
        run_label,
        *spec.args,
    ]
    started = time.perf_counter()
    completed = subprocess.run(
        cmd,
        cwd=str(REPO_ROOT),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )
    elapsed = time.perf_counter() - started
    if not summary_path.exists():
        raise RuntimeError(
            f"Component {spec.key!r} exited without writing {summary_path}.\n"
            f"stdout tail:\n{completed.stdout[-2000:]}\n"
            f"stderr tail:\n{completed.stderr[-2000:]}"
        )

    summary = _load_json(summary_path)
    if completed.returncode != 0:
        raise RuntimeError(
            f"Component {spec.key!r} failed with return code {completed.returncode}.\n"
            f"summary_path={summary_path}\n"
            f"stdout tail:\n{completed.stdout[-2000:]}\n"
            f"stderr tail:\n{completed.stderr[-2000:]}"
        )

    return {
        "key": spec.key,
        "run_label": run_label,
        "summary_path": str(summary_path),
        "reused_existing": False,
        "subprocess_returncode": int(completed.returncode),
        "subprocess_seconds": elapsed,
        "summary": summary,
    }


def _case_status_ok(summary: dict[str, Any], key: str) -> None:
    try:
        case = summary["timing_cases"][key]
    except KeyError as exc:
        raise KeyError(f"Expected timing case {key!r} in summary {summary.get('label')!r}.") from exc
    status = case.get("status")
    if status != "ok":
        raise RuntimeError(
            f"Timing case {key!r} in summary {summary.get('label')!r} returned status={status!r}."
        )


def _read_primary_seconds(summary: dict[str, Any], key: str) -> float:
    _case_status_ok(summary, key)
    return float(summary["timing_cases"][key]["primary_seconds"])


def _build_row_records(
    *,
    legacy_cold: dict[str, Any],
    legacy_prewarm: dict[str, Any],
    cupy_host_and_device_cold: dict[str, Any],
    cupy_host_prewarm: dict[str, Any],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for lane in LANE_ORDER:
        dims = [int(v) for v in SIZE_SPECS[lane].shape]
        for startup in STARTUP_ORDER:
            legacy_summary = legacy_cold if startup == "cold" else legacy_prewarm
            host_summary = cupy_host_and_device_cold if startup == "cold" else cupy_host_prewarm
            for energy_label in ENERGY_ORDER:
                for rotation_label, fragments, rotation_spec in ROTATION_ORDER:
                    fragment = fragments[energy_label]
                    legacy_key = f"core_shell_{lane}_{fragment}_host_cyrsoxs"
                    host_key = f"core_shell_{lane}_{fragment}_host_tensor_coeff"
                    device_key = f"core_shell_{lane}_{fragment}_device_tensor_coeff"

                    legacy_primary = _read_primary_seconds(legacy_summary, legacy_key)
                    host_primary = _read_primary_seconds(host_summary, host_key)
                    host_speedup = None if host_primary == 0.0 else legacy_primary / host_primary

                    device_primary = None
                    device_speedup_vs_legacy_prewarm = None
                    if startup == "pre-warm":
                        device_primary = _read_primary_seconds(cupy_host_and_device_cold, device_key)
                        device_speedup_vs_legacy_prewarm = (
                            None if device_primary == 0.0 else legacy_primary / device_primary
                        )

                    rows.append(
                        {
                            "lane": lane,
                            "voxel_dims": dims,
                            "startup": startup,
                            "energy_label": energy_label,
                            "rotation_label": rotation_label,
                            "rotation_spec": rotation_spec,
                            "legacy_cyrsoxs_primary_seconds": legacy_primary,
                            "cupy_rsoxs_host_primary_seconds": host_primary,
                            "cupy_rsoxs_host_speedup_vs_legacy": host_speedup,
                            "cupy_rsoxs_device_primary_seconds": device_primary,
                            "cupy_rsoxs_device_speedup_vs_legacy_prewarm": (
                                device_speedup_vs_legacy_prewarm
                            ),
                            "device_comparison_note": (
                                "Compared against the matching legacy pre-warm row."
                                if startup == "pre-warm"
                                else "Not reported on cold rows."
                            ),
                        }
                    )
    return rows


def _fmt_time(value: float | None) -> str:
    if value is None:
        return "—"
    return f"{float(value):.3f} s"


def _fmt_speedup(value: float | None) -> str:
    if value is None:
        return "—"
    return f"{float(value):.1f}x"


def _display_rows(rows: list[dict[str, Any]]) -> list[dict[str, str]]:
    return [
        {
            "Lane": row["lane"],
            "Voxel dims": str(tuple(row["voxel_dims"])),
            "Startup": row["startup"],
            "Energy": row["energy_label"],
            "Rotation": row["rotation_label"],
            "Legacy cyrsoxs": _fmt_time(row["legacy_cyrsoxs_primary_seconds"]),
            "cupy-rsoxs host": _fmt_time(row["cupy_rsoxs_host_primary_seconds"]),
            "Host speedup vs legacy": _fmt_speedup(row["cupy_rsoxs_host_speedup_vs_legacy"]),
            "cupy-rsoxs device": _fmt_time(row["cupy_rsoxs_device_primary_seconds"]),
            "Device speedup vs legacy pre-warm": _fmt_speedup(
                row["cupy_rsoxs_device_speedup_vs_legacy_prewarm"]
            ),
        }
        for row in rows
    ]


def _write_tsv(path: Path, rows: list[dict[str, Any]]) -> None:
    display_rows = _display_rows(rows)
    headers = list(display_rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=headers, delimiter="\t")
        writer.writeheader()
        writer.writerows(display_rows)


def _render_png(path: Path, rows: list[dict[str, Any]], *, title_label: str) -> None:
    display_rows = _display_rows(rows)
    headers = list(display_rows[0].keys())
    cell_text = [[row[h] for h in headers] for row in display_rows]

    fig_height = max(8.5, 4.0 + 0.25 * len(rows))
    fig, ax = plt.subplots(figsize=(26, fig_height), dpi=220)
    ax.axis("off")

    table = ax.table(
        cellText=cell_text,
        colLabels=headers,
        cellLoc="center",
        colLoc="center",
        loc="upper center",
        bbox=[0.0, 0.12, 1.0, 0.86],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.55)

    header_color = "#1f3b4d"
    row_alt = "#f4f7fb"
    lane_break = "#e5eef7"
    for (row_idx, _col_idx), cell in table.get_celld().items():
        if row_idx == 0:
            cell.set_facecolor(header_color)
            cell.set_text_props(color="white", weight="bold")
            cell.set_height(cell.get_height() * 1.15)
            continue
        data = rows[row_idx - 1]
        base_color = (
            lane_break
            if data["energy_label"] == "single"
            and data["rotation_label"] == "no rotation"
            and data["startup"] == "cold"
            else ("white" if row_idx % 2 else row_alt)
        )
        cell.set_facecolor(base_color)
        cell.set_edgecolor("#90a4b8")

    for row_idx, data in enumerate(rows, start=1):
        if data["startup"] == "pre-warm":
            table[(row_idx, headers.index("Startup"))].set_text_props(weight="bold")
        table[(row_idx, headers.index("Host speedup vs legacy"))].set_text_props(
            color="#7a3e00", weight="bold"
        )
        if data["cupy_rsoxs_device_primary_seconds"] is not None:
            table[(row_idx, headers.index("cupy-rsoxs device"))].set_text_props(
                color="#0b5d1e", weight="bold"
            )
            table[(row_idx, headers.index("Device speedup vs legacy pre-warm"))].set_text_props(
                color="#0b5d1e", weight="bold"
            )

    fig.suptitle(
        "Principal Cross-Backend Primary Time Comparison: legacy cyrsoxs vs cupy-rsoxs host/device",
        fontsize=18,
        fontweight="bold",
        y=0.995,
    )
    fig.text(
        0.5,
        0.065,
        (
            "Single-energy and triple-energy CoreShell lanes. Host speedups use the matching "
            "legacy startup state. Device columns appear on pre-warm rows only, because the "
            "cupy-rsoxs device lane does not meaningfully participate in the cold/pre-warm "
            "split and is compared against the matching legacy pre-warm row."
        ),
        ha="center",
        va="center",
        fontsize=10,
    )
    fig.text(
        0.5,
        0.03,
        f"Run label: {title_label}",
        ha="center",
        va="center",
        fontsize=9,
        color="#44515c",
    )
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def _write_summary(path: Path, summary: dict[str, Any]) -> None:
    path.write_text(json.dumps(summary, indent=2, default=_json_default) + "\n")


def _load_combined_summary(path: Path) -> dict[str, Any]:
    summary = _load_json(path)
    if "rows" not in summary:
        raise RuntimeError(f"Combined summary at {path} does not contain rows.")
    return summary


def run_comparison(args: argparse.Namespace) -> int:
    if not has_visible_gpu():
        raise SystemExit("No visible NVIDIA GPU found for the principal backend comparison study.")

    run_label = args.label or _timestamp()
    run_dir = OUT_ROOT / run_label
    run_dir.mkdir(parents=True, exist_ok=True)
    summary_path = run_dir / SUMMARY_NAME
    table_path = run_dir / TABLE_NAME
    figure_path = run_dir / FIGURE_NAME

    if args.plot_only:
        summary = _load_combined_summary(summary_path)
        _write_tsv(table_path, summary["rows"])
        _render_png(figure_path, summary["rows"], title_label=run_label)
        print(f"Wrote table to {table_path}", flush=True)
        print(f"Wrote PNG to {figure_path}", flush=True)
        return 0

    component_runs: dict[str, dict[str, Any]] = {}
    print("Running principal cross-backend comparison panel...", flush=True)
    for spec in COMPONENT_SPECS:
        component = _run_component(spec, label=run_label, force_rerun=args.force_rerun_subruns)
        component_runs[spec.key] = component
        action = "Reused" if component.get("reused_existing") else "Ran"
        print(f"{action} {spec.key} -> {component['summary_path']}", flush=True)

    rows = _build_row_records(
        legacy_cold=component_runs["legacy_cold"]["summary"],
        legacy_prewarm=component_runs["legacy_prewarm"]["summary"],
        cupy_host_and_device_cold=component_runs["cupy_host_and_device_cold"]["summary"],
        cupy_host_prewarm=component_runs["cupy_host_prewarm"]["summary"],
    )
    summary = {
        "label": run_label,
        "created_utc": _timestamp(),
        "python_executable": sys.executable,
        "panel_definition": {
            "lanes": list(LANE_ORDER),
            "lane_shapes": {label: list(SIZE_SPECS[label].shape) for label in LANE_ORDER},
            "startup_states": list(STARTUP_ORDER),
            "energy_panels": list(ENERGY_ORDER),
            "rotations": [label for label, _fragment, _spec in ROTATION_ORDER],
            "device_rows_reported_on": "pre-warm only",
        },
        "component_runs": {
            key: {
                subkey: value
                for subkey, value in component.items()
                if subkey != "summary"
            }
            for key, component in component_runs.items()
        },
        "rows": rows,
    }
    _write_summary(summary_path, summary)
    _write_tsv(table_path, rows)
    _render_png(figure_path, rows, title_label=run_label)
    print(f"Wrote summary to {summary_path}", flush=True)
    print(f"Wrote table to {table_path}", flush=True)
    print(f"Wrote PNG to {figure_path}", flush=True)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Principal cross-backend primary-time comparison for backend development. "
            "Runs the fixed single-energy and triple-energy CoreShell panel across "
            "legacy cyrsoxs, cupy-rsoxs host cold, cupy-rsoxs host pre-warm, and "
            "cupy-rsoxs device, then writes a combined summary, TSV, and PNG table."
        )
    )
    parser.add_argument(
        "--label",
        default=None,
        help="Output subdirectory label under test-reports/core-shell-backend-performance-dev.",
    )
    parser.add_argument(
        "--plot-only",
        action="store_true",
        help="Regenerate the TSV and PNG from an existing combined summary without rerunning sub-panels.",
    )
    parser.add_argument(
        "--force-rerun-subruns",
        action="store_true",
        help="Force rerunning the four component timing summaries even if their summary.json files already exist.",
    )
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    return run_comparison(args)


if __name__ == "__main__":
    raise SystemExit(main())
