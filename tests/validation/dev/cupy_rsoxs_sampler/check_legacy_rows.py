"""Compare compact complex basis rows and complete public results across a revision pair."""

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "cupy_rsoxs_qmax"))
from qmax_reference import bit_matches, compare
from run_public_pair import load_sample, save_sample, source_fingerprint


CASES = {
    "native": ((5, 17, 18), 0, {}),
    "native-odd-hann": ((5, 17, 19), 1, {}),
    "z-sum-fill": ((5, 17, 19), 1, {"virtual_z_pad_to": 12, "virtual_pad_material": 1}),
    "z-residues-fill": ((5, 17, 19), 1, {"virtual_z_pad_to": 20, "virtual_z_method": "residues",
                                        "virtual_pad_material": 2}),
    "xyz-factored-window": ((5, 17, 19), 1, {"virtual_3d_pad_to": (16, 32, 36), "virtual_pad_material": 1,
                                            "q_max": 0.65, "result_layout": "i_a", "result_chunk_size": 2}),
    "identity": ((5, 17, 19), 1, {"virtual_3d_pad_to": (5, 17, 19)}),
    "singleton": ((1, 17, 20), 0, {}),
    "compact-c2c": ((5, 17, 19), 1, {"energy_basis_mode": "compact_c2c", "execution_path": "tensor_coeff"}),
    "direct-r2c": ((5, 17, 19), 0, {"energy_basis_mode": "direct_r2c", "execution_path": "direct_polarization"}),
}


def worker(checkout, out):
    import cupy as cp
    import NRSS
    from NRSS.backends import _cupy_energy_basis
    from NRSS.backends._cupy_virtual_xyz import XYZ_FACTORED_XY_KEY
    from tests.smoke.test_cupy_virtual_z import _morphology, _specimen

    assert Path(NRSS.__file__).resolve() == checkout / "src/NRSS/__init__.py"
    out.mkdir()
    original = _cupy_energy_basis.build_energy_basis
    reports = {}
    for name, (shape, window, options) in CASES.items():
        material = _specimen(shape)
        # Both oriented and isotropic rows carry nonzero optical contrast.
        material[2].opt_constants = {energy: [0.001, 0.0003] * 2 for energy in material[2].energies}
        morph = _morphology(material, window, **options)
        if name == "xyz-factored-window":
            morph._backend_runtime_state[XYZ_FACTORED_XY_KEY] = True
        if name == "direct-r2c":
            morph.EAngleRotation = [0., 0., 0.]
        captures = []

        def capture(*args, **kwargs):
            basis = original(*args, **kwargs)
            captures.append((cp.asnumpy(basis.indices), cp.asnumpy(basis.values), cp.asnumpy(basis.coefficients)))
            return basis

        try:
            with patch.object(_cupy_energy_basis, "build_energy_basis", capture):
                morph._set_private_backend_timing_segments(("energy",))
                product = morph.run(stdout=False, stderr=False)
            reports[name] = morph.backend_timings["execution"]
            if len(captures) != 1:
                raise AssertionError(f"{name}: expected one compact basis, got {len(captures)}: {reports[name]}")
            (indices, values, coefficients), = captures
            np.savez(out / f"{name}-basis.npz", indices=indices, values=values, coefficients=coefficients)
            save_sample(product, out / f"{name}-result.npz")
        finally:
            morph.release_runtime()
    metadata = {"source": str(NRSS.__file__), "source_sha256": source_fingerprint(checkout),
                "sha": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True, cwd=checkout).strip(),
                "execution": reports}
    (out / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")


def verify(out):
    parent, candidate = out / "parent", out / "candidate"
    reports = [json.loads((root / "metadata.json").read_text())["execution"] for root in (parent, candidate)]
    results = {}
    for name in CASES:
        if reports[0][name] != reports[1][name]:
            raise AssertionError(f"{name}: execution signature changed")
        with (np.load(parent / f"{name}-basis.npz", allow_pickle=False) as a,
              np.load(candidate / f"{name}-basis.npz", allow_pickle=False) as b):
            fields = {}
            for key in ("indices", "values", "coefficients"):
                mismatch = np.argwhere(~bit_matches(a[key], b[key]))
                if mismatch.size:
                    first = tuple(mismatch[0])
                    raise AssertionError(f"{name}/{key}: {len(mismatch)} mismatches, first at {first}, "
                                         f"{a[key][first].tobytes().hex()} -> {b[key][first].tobytes().hex()}")
                fields[key] = a[key].size
        panels = compare(*(load_sample(root / f"{name}-result.npz") for root in (parent, candidate)))
        results[name] = {"basis_samples_checked": fields, "bit_mismatches": 0, "public_result": panels}
    (out / "summary.json").write_text(json.dumps(results, indent=2) + "\n")
    print(f"{len(results)} paired executions, {sum(r['basis_samples_checked']['values'] for r in results.values()):,} "
          "basis coefficients and all public results agree bit for bit.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--parent-root", type=Path, required=True)
    parser.add_argument("--candidate-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.worker:
        worker(args.parent_root, args.out)
        return
    args.out.mkdir(parents=True)
    for label, checkout in (("parent", args.parent_root), ("candidate", args.candidate_root)):
        with (args.out / f"{label}.log").open("w") as log:
            subprocess.run(
                [sys.executable, "-B", str(Path(__file__).resolve()), "--worker",
                 "--parent-root", str(checkout), "--candidate-root", str(checkout), "--out", str(args.out / label)],
                check=True, cwd=checkout, stdout=log, stderr=subprocess.STDOUT,
                env={**os.environ, "PYTHONPATH": f"{checkout / 'src'}:{checkout}"},
            )
    verify(args.out)


if __name__ == "__main__":
    main()
