"""Install this ticket packet into a clean or previously staged NRSS checkout."""
import argparse
import hashlib
from pathlib import Path
import shutil
import subprocess

# SHA-256 of files that were present in the first registration packet and
# whose contents changed in this revision. Preserve subsequent local edits.
V1 = {
    "optimization/cupy_rsoxs/fft_registration/EVIDENCE.md": "c0afa332bde221b95fddc5042d7feaf70984da5942ad5f4f1cc6a1fd55eb4fa8",
    "optimization/cupy_rsoxs/fft_registration/README_FIRST.md": "baed338dda37e62f5762a0d0f00a0d5f0f8b4dfa813c7e6d4c6c55e97e84287c",
    "optimization/cupy_rsoxs/fft_registration/RG01-grid.md": "4e64e1aa012307f54fe95e1da80d7fa641ae7b700fb83750cf03555614339499",
    "optimization/cupy_rsoxs/fft_registration/RG02-regression.md": "1bb64c872de40fadbf17740db862d04b8cf8f7e2873e3c4cbb59744f5e4d0c8f",
    "tests/validation/test_ewald_fft_registration.py": "d0adbcff5d797e517d79d1ab19ca6a22aa6dba4797b8298215a17cde322e26bd",
}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    args = parser.parse_args()
    repo = args.repo.resolve()
    reg = Path(__file__).resolve().parent
    stage = reg.parents[2]
    if repo == stage:
        parser.error("Extract the archive to a separate staging directory first.")
    if not (repo / "tests/validation/test_bragg_3d_lattice.py").is_file():
        parser.error("Pass an NRSS checkout containing the maintained 3D Bragg suite.")

    patch = str(reg / "bragg-tests.patch")
    git = ["git","-C",str(repo),"apply"]
    check = subprocess.run(git+["--check",patch], capture_output=True, text=True)
    apply_patch = check.returncode == 0
    if not apply_patch:
        reverse = subprocess.run(git+["--reverse","--check",patch],capture_output=True)
        if reverse.returncode:
            parser.error("Reconcile the Bragg integration patch with local changes.\n"+check.stderr)

    files = [f for top in (reg, stage / "tests") for f in top.rglob("*")
             if f.is_file() and "__pycache__" not in f.parts and f.suffix != ".pyc"]
    pending = []
    conflicts = []
    for f in sorted(files):
        relative = f.relative_to(stage)
        dest = repo / relative
        if dest.exists():
            current = digest(dest)
            if current == digest(f):
                continue
            if current != V1.get(relative.as_posix()):
                conflicts.append(str(relative))
                continue
        pending.append((f,dest))
    if conflicts:
        parser.error("These files have local edits. Merge them with the staged copies, then rerun.\n"
                     +"\n".join(conflicts))
    for source,dest in pending:
        dest.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(source,dest)
    if apply_patch:
        subprocess.run(git+[patch],check=True)
    print(f"Installed {len(pending)} packet files; Bragg patch "
          +("applied." if apply_patch else "already present."))


if __name__=="__main__":
    main()
