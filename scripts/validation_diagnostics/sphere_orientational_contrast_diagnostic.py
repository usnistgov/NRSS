"""Opt-in plot generator for the official sphere orientational-contrast validation."""

from __future__ import annotations

import os
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tests.validation.test_sphere_orientational_contrast_scaling import (  # noqa: E402
    PLOT_DIR,
    evaluate_orientational_ratio_rows,
    write_orientational_validation_artifacts,
    _has_visible_gpu,
)


def main() -> int:
    if not _has_visible_gpu():
        print("No visible NVIDIA GPU found; orientational contrast diagnostic skipped.")
        return 0

    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
    rows = evaluate_orientational_ratio_rows()
    outputs = write_orientational_validation_artifacts(rows)
    non_ref_rows = [row for row in rows if row["series"] != "reference"]
    worst = max(non_ref_rows, key=lambda row: float(row["rel_err"]))

    print(f"Wrote orientational contrast artifacts under: {PLOT_DIR}")
    for out in outputs:
        print(f"  - {out}")
    print(
        "Worst case: "
        f"family={worst['family']} "
        f"scenario={worst['label']} "
        f"expected={float(worst['expected_ratio']):.9f} "
        f"observed={float(worst['observed_ratio']):.9f} "
        f"rel={float(worst['rel_err']):.6f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
