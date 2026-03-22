#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ast
import json
import os
import re
import shutil
import subprocess
import sys
import time
import zipfile
from collections import Counter
from dataclasses import dataclass
from pathlib import Path


SUMMARY_COUNT_ORDER = (
    "passed",
    "failed",
    "errors",
    "skipped",
    "deselected",
    "xfailed",
    "xpassed",
)
MAX_ROW_ATTEMPTS = 2


@dataclass(frozen=True)
class PhysicsRow:
    index: int
    test_file: Path
    test_name: str
    markers: str
    summary: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run markdown-row physics validation tests sequentially and optionally "
            "harvest the generated PNG graphical abstracts."
        )
    )
    parser.add_argument("--repo-root", required=True, type=Path)
    parser.add_argument("--catalog", required=True, type=Path)
    parser.add_argument("--harvest-root", required=True, type=Path)
    parser.add_argument("--zip-path", required=True, type=Path)
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument("--nrss-backend", default=None)
    return parser.parse_args()


def slugify(text: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9]+", "_", text.strip().lower()).strip("_")
    return slug or "item"


def natural_key(text: str) -> list[object]:
    return [int(token) if token.isdigit() else token.lower() for token in re.split(r"(\d+)", text)]


def parse_catalog(path: Path) -> list[PhysicsRow]:
    rows: list[PhysicsRow] = []
    for index, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        parts = line.split("\t")
        test_file = Path(parts[0])
        test_name = parts[1] if len(parts) > 1 else ""
        markers = parts[2] if len(parts) > 2 else ""
        summary = parts[3] if len(parts) > 3 else ""
        rows.append(
            PhysicsRow(
                index=index,
                test_file=test_file,
                test_name=test_name,
                markers=markers,
                summary=summary,
            )
        )
    return rows


def _eval_path_expr(node: ast.AST, repo_root: Path) -> Path | str:
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    if isinstance(node, ast.Name) and node.id == "REPO_ROOT":
        return repo_root
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Div):
        left = _eval_path_expr(node.left, repo_root)
        right = _eval_path_expr(node.right, repo_root)
        return Path(left) / str(right)
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "Path":
        if len(node.args) != 1:
            raise ValueError("Path() plot expression must have exactly one argument.")
        return Path(_eval_path_expr(node.args[0], repo_root))
    raise ValueError(f"Unsupported plot-path expression: {ast.dump(node, include_attributes=False)}")


def extract_plot_dirs(test_file: Path, repo_root: Path) -> list[Path]:
    tree = ast.parse(test_file.read_text(encoding="utf-8"), filename=str(test_file))
    discovered: list[Path] = []

    def add_path(value: ast.AST) -> None:
        resolved = _eval_path_expr(value, repo_root)
        path = Path(resolved)
        if not path.is_absolute():
            path = repo_root / path
        discovered.append(path)

    for node in tree.body:
        if isinstance(node, ast.Assign):
            targets = [target.id for target in node.targets if isinstance(target, ast.Name)]
            value = node.value
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            targets = [node.target.id]
            value = node.value
        else:
            continue

        if not value:
            continue
        if "PLOT_DIR" in targets:
            add_path(value)
        if "PLOT_DIRS" in targets and isinstance(value, (ast.List, ast.Tuple, ast.Set)):
            for item in value.elts:
                add_path(item)

    unique_paths: list[Path] = []
    seen: set[Path] = set()
    for path in discovered:
        resolved = path.resolve()
        if resolved not in seen:
            seen.add(resolved)
            unique_paths.append(resolved)
    return unique_paths


def snapshot_png_state(plot_dirs: list[Path]) -> dict[Path, tuple[int, int]]:
    state: dict[Path, tuple[int, int]] = {}
    for plot_dir in plot_dirs:
        if not plot_dir.exists():
            continue
        for png_path in plot_dir.rglob("*.png"):
            try:
                stat = png_path.stat()
            except FileNotFoundError:
                continue
            state[png_path.resolve()] = (stat.st_mtime_ns, stat.st_size)
    return state


def parse_summary_counts(output: str) -> Counter[str]:
    counts: Counter[str] = Counter()
    for raw_line in reversed(output.splitlines()):
        line = raw_line.strip().strip("=")
        if " in " not in line:
            continue
        counts_part = line.split(" in ", 1)[0].strip()
        if not re.search(
            r"\b(passed|failed|error|errors|skipped|xfailed|xpassed|deselected|warning|warnings)\b",
            counts_part,
        ):
            continue
        for part in counts_part.split(","):
            match = re.match(r"^\s*(\d+)\s+([A-Za-z]+)\s*$", part)
            if not match:
                continue
            count = int(match.group(1))
            label = match.group(2).lower()
            if label in {"warning", "warnings"}:
                counts["warnings"] += count
            elif label in {"error", "errors"}:
                counts["errors"] += count
            else:
                counts[label] += count
        if counts:
            return counts
    return counts


def render_summary_line(total_counts: Counter[str], duration_s: float) -> str:
    summary_parts: list[str] = []
    for label in SUMMARY_COUNT_ORDER:
        count = total_counts.get(label, 0)
        if count <= 0:
            continue
        noun = "error" if label == "errors" and count == 1 else label
        noun = noun[:-1] if noun.endswith("s") and count == 1 else noun
        summary_parts.append(f"{count} {noun}")
    warning_count = total_counts.get("warnings", 0)
    if warning_count > 0:
        noun = "warning" if warning_count == 1 else "warnings"
        summary_parts.append(f"{warning_count} {noun}")

    if not summary_parts:
        payload = "no tests ran"
    else:
        payload = ", ".join(summary_parts)
    return f"================= {payload} in {duration_s:.2f}s =================="


def run_pytest_row(
    repo_root: Path,
    row: PhysicsRow,
    write_plots: bool,
    nrss_backend: str | None = None,
) -> tuple[int, Counter[str]]:
    env = os.environ.copy()
    env.setdefault("CUDA_VISIBLE_DEVICES", "0")
    if nrss_backend is not None and nrss_backend.strip():
        env["NRSS_BACKEND"] = nrss_backend.strip()
    if write_plots:
        env["NRSS_WRITE_VALIDATION_PLOTS"] = "1"
    else:
        env.pop("NRSS_WRITE_VALIDATION_PLOTS", None)

    nodeid = f"{row.test_file.as_posix()}::{row.test_name}"
    cmd = [sys.executable, "-m", "pytest", nodeid, "-v"]
    last_rc = 0
    last_counts: Counter[str] = Counter()

    for attempt in range(1, MAX_ROW_ATTEMPTS + 1):
        proc = subprocess.run(
            cmd,
            cwd=str(repo_root),
            env=env,
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        if proc.stdout:
            sys.stdout.write(proc.stdout)
            if not proc.stdout.endswith("\n"):
                sys.stdout.write("\n")
        sys.stdout.flush()

        last_rc = proc.returncode
        last_counts = parse_summary_counts(proc.stdout or "")
        if proc.returncode == 0:
            return proc.returncode, last_counts

        if attempt < MAX_ROW_ATTEMPTS:
            print(
                f"physics row retry: {row.test_name} "
                f"(attempt {attempt + 1}/{MAX_ROW_ATTEMPTS} after rc={proc.returncode})"
            )
            sys.stdout.flush()

    return last_rc, last_counts


def harvest_row_pngs(
    row: PhysicsRow,
    total_rows: int,
    before_state: dict[Path, tuple[int, int]],
    plot_dirs: list[Path],
    harvest_root: Path,
) -> list[dict[str, str]]:
    after_state = snapshot_png_state(plot_dirs)
    changed = [path for path, meta in after_state.items() if before_state.get(path) != meta]
    changed.sort(key=lambda path: (after_state[path][0], natural_key(path.name), natural_key(str(path))))

    harvested: list[dict[str, str]] = []
    row_width = max(2, len(str(total_rows)))
    row_slug = slugify(row.test_name)

    for item_index, source in enumerate(changed, start=1):
        destination_name = f"{row.index:0{row_width}d}_{item_index:02d}_{row_slug}__{source.name}"
        destination = harvest_root / destination_name
        shutil.copy2(source, destination)
        harvested.append(
            {
                "row_index": f"{row.index:0{row_width}d}",
                "item_index": f"{item_index:02d}",
                "test_file": row.test_file.as_posix(),
                "test_name": row.test_name,
                "source_path": source.as_posix(),
                "harvested_path": destination.as_posix(),
            }
        )

    return harvested


def write_manifests(harvest_root: Path, harvested_rows: list[dict[str, str]]) -> list[Path]:
    tsv_path = harvest_root / "graphical_abstracts_manifest.tsv"
    json_path = harvest_root / "graphical_abstracts_manifest.json"

    header = ("row_index", "item_index", "test_file", "test_name", "source_path", "harvested_path")
    with tsv_path.open("w", encoding="utf-8") as handle:
        handle.write("\t".join(header) + "\n")
        for row in harvested_rows:
            handle.write("\t".join(row[key] for key in header) + "\n")

    json_path.write_text(json.dumps(harvested_rows, indent=2) + "\n", encoding="utf-8")
    return [tsv_path, json_path]


def build_zip(zip_path: Path, harvest_root: Path) -> None:
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for item in sorted(harvest_root.rglob("*"), key=lambda path: natural_key(str(path.relative_to(harvest_root.parent)))):
            if item.is_dir():
                continue
            archive.write(item, arcname=item.relative_to(harvest_root.parent))


def main() -> int:
    args = parse_args()
    repo_root = args.repo_root.resolve()
    catalog_path = args.catalog.resolve()
    rows = parse_catalog(catalog_path)

    total_counts: Counter[str] = Counter()
    any_fail = False
    start = time.perf_counter()
    harvested_rows: list[dict[str, str]] = []

    if not args.no_plots:
        args.harvest_root.mkdir(parents=True, exist_ok=True)

    for row in rows:
        test_file = (repo_root / row.test_file).resolve()
        plot_dirs: list[Path] = []
        if not args.no_plots:
            try:
                plot_dirs = extract_plot_dirs(test_file, repo_root)
            except Exception as exc:
                print(f"graphical-abstract harvest warning: {row.test_name}: {exc}")
                sys.stdout.flush()
        before_state = snapshot_png_state(plot_dirs) if plot_dirs else {}

        rc, counts = run_pytest_row(
            repo_root,
            row,
            write_plots=not args.no_plots,
            nrss_backend=args.nrss_backend,
        )
        total_counts.update(counts)
        if rc != 0:
            any_fail = True

        if plot_dirs:
            harvested_rows.extend(
                harvest_row_pngs(
                    row=row,
                    total_rows=len(rows),
                    before_state=before_state,
                    plot_dirs=plot_dirs,
                    harvest_root=args.harvest_root,
                )
            )

    duration_s = time.perf_counter() - start
    print(render_summary_line(total_counts, duration_s))

    if not args.no_plots:
        write_manifests(args.harvest_root, harvested_rows)
        build_zip(args.zip_path.resolve(), args.harvest_root)

    return 1 if any_fail else 0


if __name__ == "__main__":
    raise SystemExit(main())
