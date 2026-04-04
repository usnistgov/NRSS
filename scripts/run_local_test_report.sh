#!/usr/bin/env bash
set -u
set -o pipefail

usage() {
  cat <<'EOF'
Usage:
  scripts/run_local_test_report.sh [options]

Options:
  -e, --env NAME            Conda environment name (default: $NRSS_TEST_ENV or nrss-dev)
  -r, --report-root PATH    Report root directory (default: test-reports)
  --no-plots                Disable default physics plot harvesting and zip creation.
  --cyrsoxs-cli-dir PATH    Prepend PATH to CLI lookup inside each conda-run step.
  --cyrsoxs-pybind-dir PATH Prepend PATH to PYTHONPATH inside each conda-run step.
  --nrss-backend NAME       Export NRSS_BACKEND=NAME inside each test step.
  --cmd "COMMAND"           Add a test command to run in conda env.
                            Can be passed multiple times. Custom commands run after the
                            default suite unless --skip-defaults is set.
  --skip-defaults           Skip the standard report suite and run only explicitly
                            provided --cmd commands.
  --repeat N                Repeat each explicit --cmd command N times (default: 1).
  --stop-on-fail            Stop after first failing step.
  -h, --help                Show this help.

Environment overrides:
  NRSS_TEST_CYRSOXS_CLI_DIR     Same as --cyrsoxs-cli-dir.
  NRSS_TEST_CYRSOXS_PYBIND_DIR  Same as --cyrsoxs-pybind-dir.
  NRSS_TEST_BACKEND             Same as --nrss-backend.
  NRSS_TEST_ENV                 Same as --env.

Behavior:
  By default, the standard local report runs four steps:
    1. Environment Snapshot
    2. Smoke Tests (CPU Fast)
    3. Smoke Tests (GPU)
    4. Physics Validation Tests
  Physics Validation Tests also harvest any generated graphical-abstract PNGs
  into the per-run report directory and write graphical-abstracts.zip.
  Any explicit --cmd commands are appended afterward.
  Use --skip-defaults to run only the explicit commands.
  Use --repeat N to repeat each explicit command N times.

Examples:
  scripts/run_local_test_report.sh
  NRSS_TEST_ENV=mar2025 scripts/run_local_test_report.sh
  scripts/run_local_test_report.sh -e nrss-dev \\
    --cyrsoxs-cli-dir /path/to/cyrsoxs/build \\
    --cyrsoxs-pybind-dir /path/to/cyrsoxs/build-pybind
  scripts/run_local_test_report.sh --nrss-backend cyrsoxs
  NRSS_TEST_CYRSOXS_CLI_DIR=/path/to/cyrsoxs/build \\
    NRSS_TEST_CYRSOXS_PYBIND_DIR=/path/to/cyrsoxs/build-pybind \\
    scripts/run_local_test_report.sh -e nrss-dev
  scripts/run_local_test_report.sh --cmd "python -m pytest -m 'not slow' -q"
  scripts/run_local_test_report.sh --skip-defaults --repeat 10 \\
    --cmd "python -m pytest tests/validation/test_analytical_2d_disk_form_factor.py -q"
  scripts/run_local_test_report.sh --cmd "python -m pytest tests/smoke -m gpu -q"
  scripts/run_local_test_report.sh -e mar2025 --cmd "python -m pytest tests/validation -q"
  scripts/run_local_test_report.sh --no-plots
EOF
}

ENV_NAME="${NRSS_TEST_ENV:-nrss-dev}"
REPORT_ROOT="test-reports"
STOP_ON_FAIL=0
SKIP_DEFAULTS=0
CUSTOM_REPEAT_COUNT=1
WRITE_PHYSICS_PLOTS=1
CYRSOXS_CLI_DIR="${NRSS_TEST_CYRSOXS_CLI_DIR:-${NRSS_TEST_PATH_PREPEND:-}}"
CYRSOXS_PYBIND_DIR="${NRSS_TEST_CYRSOXS_PYBIND_DIR:-${NRSS_TEST_PYTHONPATH_PREPEND:-}}"
NRSS_BACKEND_NAME="${NRSS_TEST_BACKEND:-${NRSS_BACKEND:-}}"
declare -a TEST_CMDS=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    -e|--env)
      ENV_NAME="${2:-}"
      shift 2
      ;;
    -r|--report-root)
      REPORT_ROOT="${2:-}"
      shift 2
      ;;
    --no-plots)
      WRITE_PHYSICS_PLOTS=0
      shift
      ;;
    --cyrsoxs-cli-dir)
      CYRSOXS_CLI_DIR="${2:-}"
      shift 2
      ;;
    --cyrsoxs-pybind-dir)
      CYRSOXS_PYBIND_DIR="${2:-}"
      shift 2
      ;;
    --nrss-backend)
      NRSS_BACKEND_NAME="${2:-}"
      shift 2
      ;;
    --cmd)
      TEST_CMDS+=("${2:-}")
      shift 2
      ;;
    --skip-defaults)
      SKIP_DEFAULTS=1
      shift
      ;;
    --repeat)
      CUSTOM_REPEAT_COUNT="${2:-}"
      shift 2
      ;;
    --stop-on-fail)
      STOP_ON_FAIL=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

if ! [[ "$CUSTOM_REPEAT_COUNT" =~ ^[1-9][0-9]*$ ]]; then
  echo "--repeat expects a positive integer, got: ${CUSTOM_REPEAT_COUNT}" >&2
  exit 2
fi

if ! command -v conda >/dev/null 2>&1; then
  echo "conda is required but not found in PATH." >&2
  exit 2
fi

if ! git rev-parse --show-toplevel >/dev/null 2>&1; then
  echo "Run this script inside a git repository." >&2
  exit 2
fi

REPO_ROOT="$(git rev-parse --show-toplevel)"
cd "$REPO_ROOT" || exit 2

TIMESTAMP_UTC="$(date -u +%Y%m%dT%H%M%SZ)"
REPORT_DIR="${REPORT_ROOT%/}/${TIMESTAMP_UTC}"
mkdir -p "$REPORT_DIR"

RUN_LOG="$REPORT_DIR/run.log"
SUMMARY_MD="$REPORT_DIR/summary.md"
METADATA_TXT="$REPORT_DIR/metadata.txt"
STEPS_TSV="$REPORT_DIR/steps.tsv"
SMOKE_CATALOG_TSV="$REPORT_DIR/smoke_catalog.tsv"
PHYSICS_CATALOG_TSV="$REPORT_DIR/physics_catalog.tsv"
NRSS_RESOLUTION_TSV="$REPORT_DIR/nrss_resolution.tsv"
CYRSOXS_RESOLUTION_TSV="$REPORT_DIR/cyrsoxs_resolution.tsv"
VALIDATION_REFERENCE_MANIFEST_TSV="$REPORT_DIR/validation_reference_manifest.tsv"
GRAPHICAL_ABSTRACTS_DIR="$REPORT_DIR/graphical-abstracts"
GRAPHICAL_ABSTRACTS_ZIP="$REPORT_DIR/graphical-abstracts.zip"

: > "$RUN_LOG"
: > "$STEPS_TSV"

log() {
  local msg="$1"
  printf '%s\n' "$msg" | tee -a "$RUN_LOG"
}

slugify() {
  echo "$1" | tr '[:upper:]' '[:lower:]' | tr -cs 'a-z0-9' '_' | sed 's/^_//;s/_$//'
}

build_inner_cmd() {
  local cmd="$1"
  local inner_cmd

  inner_cmd="$cmd"
  if [[ -n "$CYRSOXS_PYBIND_DIR" ]]; then
    inner_cmd="export PYTHONPATH=\"${CYRSOXS_PYBIND_DIR}\${PYTHONPATH:+:\$PYTHONPATH}\""$'\n'"$inner_cmd"
  fi
  if [[ -n "$CYRSOXS_CLI_DIR" ]]; then
    inner_cmd="export PATH=\"${CYRSOXS_CLI_DIR}:\$PATH\""$'\n'"$inner_cmd"
  fi
  if [[ -n "$NRSS_BACKEND_NAME" ]]; then
    inner_cmd="export NRSS_BACKEND=\"${NRSS_BACKEND_NAME}\""$'\n'"$inner_cmd"
  fi

  printf '%s' "$inner_cmd"
}

write_metadata() {
  local git_sha git_branch git_dirty visible_gpus
  git_sha="$(git rev-parse HEAD)"
  git_branch="$(git rev-parse --abbrev-ref HEAD)"
  if git diff --quiet && git diff --cached --quiet; then
    git_dirty="clean"
  else
    git_dirty="dirty"
  fi
  visible_gpus="${CUDA_VISIBLE_DEVICES:-0}"

  {
    echo "timestamp_utc=$TIMESTAMP_UTC"
    echo "repo_root=$REPO_ROOT"
    echo "git_branch=$git_branch"
    echo "git_sha=$git_sha"
    echo "git_worktree=$git_dirty"
    echo "conda_env=$ENV_NAME"
    echo "cuda_visible_devices=$visible_gpus"
    echo "skip_defaults=$SKIP_DEFAULTS"
    echo "custom_repeat_count=$CUSTOM_REPEAT_COUNT"
    echo "write_physics_plots=$WRITE_PHYSICS_PLOTS"
    echo "cyrsoxs_cli_dir=$CYRSOXS_CLI_DIR"
    echo "cyrsoxs_pybind_dir=$CYRSOXS_PYBIND_DIR"
    echo "nrss_backend=$NRSS_BACKEND_NAME"
    echo "nrss_resolution_tsv=$NRSS_RESOLUTION_TSV"
    echo "cyrsoxs_resolution_tsv=$CYRSOXS_RESOLUTION_TSV"
    echo "validation_reference_manifest_tsv=$VALIDATION_REFERENCE_MANIFEST_TSV"
    echo "graphical_abstracts_dir=$GRAPHICAL_ABSTRACTS_DIR"
    echo "graphical_abstracts_zip=$GRAPHICAL_ABSTRACTS_ZIP"
    if command -v nvidia-smi >/dev/null 2>&1; then
      echo "nvidia_smi=present"
      nvidia-smi -L || true
    else
      echo "nvidia_smi=not_found"
    fi
  } > "$METADATA_TXT"
}

capture_nrss_resolution() {
  local probe_cmd inner_cmd

  probe_cmd=$'python - <<\'PY\'\nimport importlib\nimport subprocess\nfrom pathlib import Path\n\n\ndef safe(value):\n    if value is None:\n        return \"\"\n    return str(value).replace(\"\\t\", \" \").replace(\"\\n\", \" \").strip()\n\n\ndef git_provenance(path_str):\n    if not path_str:\n        return \"\", \"\"\n    try:\n        path = Path(path_str).resolve()\n    except Exception:\n        return \"\", \"\"\n\n    start = path if path.is_dir() else path.parent\n    for candidate in (start, *start.parents):\n        root = subprocess.run(\n            [\"git\", \"-C\", str(candidate), \"rev-parse\", \"--show-toplevel\"],\n            check=False,\n            stdout=subprocess.PIPE,\n            stderr=subprocess.PIPE,\n            text=True,\n            timeout=2,\n        )\n        if root.returncode != 0:\n            continue\n        root_path = root.stdout.strip()\n        branch = subprocess.run(\n            [\"git\", \"-C\", root_path, \"rev-parse\", \"--abbrev-ref\", \"HEAD\"],\n            check=False,\n            stdout=subprocess.PIPE,\n            stderr=subprocess.PIPE,\n            text=True,\n            timeout=2,\n        )\n        sha = subprocess.run(\n            [\"git\", \"-C\", root_path, \"rev-parse\", \"HEAD\"],\n            check=False,\n            stdout=subprocess.PIPE,\n            stderr=subprocess.PIPE,\n            text=True,\n            timeout=2,\n        )\n        return branch.stdout.strip(), sha.stdout.strip()\n    return \"\", \"\"\n\n\nprint(\"resolved_name\\tresolved_path\\tversion\\tgit_branch\\tgit_sha\\tstatus\")\ntry:\n    mod = importlib.import_module(\"NRSS\")\nexcept Exception as exc:\n    print(f\"NRSS\\t\\t\\t\\t\\tIMPORT_FAILED:{exc.__class__.__name__}\")\nelse:\n    path = safe(getattr(mod, \"__file__\", \"\"))\n    version = safe(getattr(mod, \"__version__\", \"\"))\n    branch, sha = git_provenance(path)\n    print(\"\\t\".join([\"NRSS\", path, version, branch, sha, \"OK\"]))\nPY'
  inner_cmd="$(build_inner_cmd "$probe_cmd")"

  if ! conda run -n "$ENV_NAME" bash -lc "$inner_cmd" > "$NRSS_RESOLUTION_TSV" 2>> "$RUN_LOG"; then
    cat > "$NRSS_RESOLUTION_TSV" <<'EOF'
resolved_name	resolved_path	version	git_branch	git_sha	status
NRSS					PROBE_FAILED
EOF
    log "NRSS resolution probe failed; see $RUN_LOG for details."
  fi
}

capture_cyrsoxs_resolution() {
  local probe_cmd inner_cmd

  probe_cmd=$'python - <<\'PY\'\nimport contextlib\nimport importlib\nimport io\nimport re\nimport shutil\nimport subprocess\nfrom pathlib import Path\n\n\ndef safe(value):\n    if value is None:\n        return \"\"\n    return str(value).replace(\"\\t\", \" \").replace(\"\\n\", \" \").strip()\n\n\ndef git_provenance(path_str):\n    if not path_str:\n        return \"\", \"\", \"\"\n    try:\n        path = Path(path_str).resolve()\n    except Exception:\n        return \"\", \"\", \"\"\n\n    start = path if path.is_dir() else path.parent\n    for candidate in (start, *start.parents):\n        try:\n            root = subprocess.run(\n                [\"git\", \"-C\", str(candidate), \"rev-parse\", \"--show-toplevel\"],\n                check=False,\n                stdout=subprocess.PIPE,\n                stderr=subprocess.PIPE,\n                text=True,\n                timeout=2,\n            )\n        except Exception:\n            continue\n        if root.returncode != 0:\n            continue\n        root_path = root.stdout.strip()\n        branch = subprocess.run(\n            [\"git\", \"-C\", root_path, \"rev-parse\", \"--abbrev-ref\", \"HEAD\"],\n            check=False,\n            stdout=subprocess.PIPE,\n            stderr=subprocess.PIPE,\n            text=True,\n            timeout=2,\n        )\n        sha = subprocess.run(\n            [\"git\", \"-C\", root_path, \"rev-parse\", \"HEAD\"],\n            check=False,\n            stdout=subprocess.PIPE,\n            stderr=subprocess.PIPE,\n            text=True,\n            timeout=2,\n        )\n        return root_path, branch.stdout.strip(), sha.stdout.strip()\n    return \"\", \"\", \"\"\n\n\ndef version_from_repo_root(root_path):\n    if not root_path:\n        return \"\"\n    cmake_path = Path(root_path) / \"CMakeLists.txt\"\n    if not cmake_path.exists():\n        return \"\"\n    try:\n        text = cmake_path.read_text(encoding=\"utf-8\", errors=\"replace\")\n    except Exception:\n        return \"\"\n    match = re.search(r\"project\\s*\\(\\s*CyRSoXS\\s+VERSION\\s+([0-9.]+)\", text, re.IGNORECASE)\n    return match.group(1) if match else \"\"\n\n\ndef version_from_banner(text):\n    for line in text.splitlines():\n        match = re.search(r\"Version\\s*:\\s*([0-9\\s.]+)\", line)\n        if match:\n            return re.sub(r\"\\s+\", \"\", match.group(1))\n    return \"\"\n\n\ndef cli_record():\n    path = safe(shutil.which(\"CyRSoXS\") or \"\")\n    if not path:\n        return (\"cli\", \"CyRSoXS\", \"\", \"\", \"\", \"\", \"NOT_FOUND\")\n\n    version = \"\"\n    try:\n        proc = subprocess.run(\n            [path, \"--version\"],\n            check=False,\n            stdout=subprocess.PIPE,\n            stderr=subprocess.PIPE,\n            text=True,\n            timeout=5,\n        )\n    except Exception:\n        proc = None\n    if proc is not None:\n        version = version_from_banner((proc.stdout or \"\") + \"\\n\" + (proc.stderr or \"\"))\n\n    root_path, branch, sha = git_provenance(path)\n    if not version:\n        version = version_from_repo_root(root_path)\n    return (\"cli\", \"CyRSoXS\", path, version, branch, sha, \"OK\")\n\n\ndef pybind_record():\n    import_name = \"CyRSoXS\"\n    try:\n        buf_out = io.StringIO()\n        buf_err = io.StringIO()\n        with contextlib.redirect_stdout(buf_out), contextlib.redirect_stderr(buf_err):\n            mod = importlib.import_module(import_name)\n    except Exception as exc:\n        status = f\"{import_name}:{exc.__class__.__name__}\"\n        return (\"pybind\", import_name, \"\", \"\", \"\", \"\", status)\n    path = safe(getattr(mod, \"__file__\", \"\"))\n    banner_text = buf_out.getvalue() + \"\\n\" + buf_err.getvalue()\n    version = safe(getattr(mod, \"__version__\", \"\")) or version_from_banner(banner_text)\n    root_path, branch, sha = git_provenance(path)\n    if not version:\n        version = version_from_repo_root(root_path)\n    return (\"pybind\", import_name, path, version, branch, sha, \"OK\")\n\n\nprint(\"component\\tresolved_name\\tresolved_path\\tversion\\tgit_branch\\tgit_sha\\tstatus\")\nfor row in (cli_record(), pybind_record()):\n    print(\"\\t\".join(safe(value) for value in row))\nPY'
  inner_cmd="$(build_inner_cmd "$probe_cmd")"

  if ! CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" conda run -n "$ENV_NAME" bash -lc "$inner_cmd" > "$CYRSOXS_RESOLUTION_TSV" 2>> "$RUN_LOG"; then
    cat > "$CYRSOXS_RESOLUTION_TSV" <<'EOF'
component	resolved_name	resolved_path	version	git_branch	git_sha	status
cli	CyRSoXS					PROBE_FAILED
pybind						PROBE_FAILED
EOF
    log "CyRSoXS resolution probe failed; see $RUN_LOG for details."
  fi
}

build_validation_reference_manifest() {
  python - "$REPO_ROOT" "$VALIDATION_REFERENCE_MANIFEST_TSV" <<'PY'
import hashlib
import pathlib
import sys

repo_root = pathlib.Path(sys.argv[1]).resolve()
out_path = pathlib.Path(sys.argv[2]).resolve()
data_root = repo_root / "tests" / "validation" / "data"

rows = []
if data_root.exists():
    for path in sorted(p for p in data_root.rglob("*") if p.is_file()):
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        relpath = path.relative_to(repo_root).as_posix()
        rows.append((relpath, str(path.stat().st_size), digest))

with out_path.open("w", encoding="utf-8") as f:
    f.write("path\tsize_bytes\tsha256\n")
    for row in rows:
        f.write("\t".join(row) + "\n")
PY
}

build_smoke_catalog() {
  python - "$SMOKE_CATALOG_TSV" <<'PY'
import ast
import pathlib
import sys

out = pathlib.Path(sys.argv[1])
test_file = pathlib.Path("tests/smoke/test_smoke.py")

if not test_file.exists():
    out.write_text("", encoding="utf-8")
    raise SystemExit(0)

tree = ast.parse(test_file.read_text(encoding="utf-8"))
rows = []
for node in tree.body:
    if isinstance(node, ast.FunctionDef) and node.name.startswith("test_"):
        doc = (ast.get_docstring(node) or "").strip().splitlines()
        summary = doc[0] if doc else ""
        markers = []
        for dec in node.decorator_list:
            # Matches patterns like @pytest.mark.gpu / @pytest.mark.cpu
            if (
                isinstance(dec, ast.Attribute)
                and isinstance(dec.value, ast.Attribute)
                and isinstance(dec.value.value, ast.Name)
                and dec.value.value.id == "pytest"
                and dec.value.attr == "mark"
            ):
                markers.append(dec.attr)
        rows.append((node.lineno, node.name, ",".join(markers), summary))

rows.sort(key=lambda r: r[0])
with out.open("w", encoding="utf-8") as f:
    for _, name, marker_csv, summary in rows:
        f.write(f"{name}\t{marker_csv}\t{summary}\n")
PY
}

build_physics_catalog() {
  python - "$PHYSICS_CATALOG_TSV" <<'PY'
import ast
import pathlib
import sys

out = pathlib.Path(sys.argv[1])
validation_dir = pathlib.Path("tests/validation")

if not validation_dir.exists():
    out.write_text("", encoding="utf-8")
    raise SystemExit(0)

rows = []
for test_file in sorted(validation_dir.glob("test_*.py")):
    tree = ast.parse(test_file.read_text(encoding="utf-8"))
    for node in tree.body:
        if not isinstance(node, ast.FunctionDef) or not node.name.startswith("test_"):
            continue
        markers = []
        for dec in node.decorator_list:
            if (
                isinstance(dec, ast.Attribute)
                and isinstance(dec.value, ast.Attribute)
                and isinstance(dec.value.value, ast.Name)
                and dec.value.value.id == "pytest"
                and dec.value.attr == "mark"
            ):
                markers.append(dec.attr)
        if "physics_validation" not in markers:
            continue
        doc = (ast.get_docstring(node) or "").strip().splitlines()
        summary = "<br>".join(line.strip() for line in doc if line.strip())
        rows.append((str(test_file), node.lineno, node.name, ",".join(markers), summary))

rows.sort(key=lambda r: (r[0], r[1]))
with out.open("w", encoding="utf-8") as f:
    for test_file, _, name, marker_csv, summary in rows:
        f.write(f"{test_file}\t{name}\t{marker_csv}\t{summary}\n")
PY
}

run_conda_step() {
  local step_name="$1"
  local cmd="$2"
  local step_index="$3"
  local slug log_file case_file start_ts end_ts duration status rc result_line cmd_oneline inner_cmd

  slug="$(slugify "$step_name")"
  log_file="$REPORT_DIR/step-$(printf '%02d' "$step_index")-${slug}.log"
  case_file="$REPORT_DIR/step-$(printf '%02d' "$step_index")-${slug}.cases.tsv"

  log ""
  log "==> [$step_index] $step_name"
  log "    Command: $cmd"
  start_ts="$(date +%s)"

  inner_cmd="$(build_inner_cmd "$cmd")"
  CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" conda run -n "$ENV_NAME" bash -lc "$inner_cmd" > "$log_file" 2>&1
  rc=$?

  end_ts="$(date +%s)"
  duration=$((end_ts - start_ts))
  if [[ $rc -eq 0 ]]; then
    status="PASS"
  else
    status="FAIL"
  fi

  result_line="$(grep -E "([0-9]+ (passed|failed|error|errors|skipped|xfailed|xpassed)|no tests ran)" "$log_file" | tail -n 1 || true)"
  if [[ -z "$result_line" ]]; then
    result_line="$(tail -n 1 "$log_file" | tr -d '\r' || true)"
  fi
  result_line="${result_line//$'\t'/ }"
  cmd_oneline="${cmd//$'\n'/ }"
  cmd_oneline="${cmd_oneline//$'\t'/ }"
  cmd_oneline="$(echo "$cmd_oneline" | tr -s ' ')"

  python - "$log_file" "$case_file" <<'PY'
import pathlib
import re
import sys

log_path = pathlib.Path(sys.argv[1])
case_path = pathlib.Path(sys.argv[2])
pat = re.compile(
    r"::(test_[A-Za-z0-9_]+)(?:\[[^\]]+\])?\s+"
    r"(PASSED|FAILED|SKIPPED|XFAILED|XPASSED|ERROR)\b"
)
seen = {}
for line in log_path.read_text(encoding="utf-8", errors="replace").splitlines():
    m = pat.search(line)
    if m:
        seen[m.group(1)] = m.group(2)

with case_path.open("w", encoding="utf-8") as f:
    for name, case_status in seen.items():
        f.write(f"{name}\t{case_status}\n")
PY

  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
    "$step_index" "$step_name" "$status" "$duration" "$log_file" "$cmd_oneline" "$result_line" "$case_file" >> "$STEPS_TSV"

  log "    Status: $status (exit=$rc, ${duration}s)"
  log "    Log: $log_file"
  if [[ -n "$result_line" ]]; then
    log "    Result: $result_line"
  fi

  return "$rc"
}

generate_summary() {
  local pass_count fail_count total git_sha git_branch
  total="$(wc -l < "$STEPS_TSV" | tr -d ' ')"
  pass_count="$(awk -F'\t' '$3=="PASS"{c++} END{print c+0}' "$STEPS_TSV")"
  fail_count="$(awk -F'\t' '$3=="FAIL"{c++} END{print c+0}' "$STEPS_TSV")"
  git_sha="$(git rev-parse --short HEAD)"
  git_branch="$(git rev-parse --abbrev-ref HEAD)"

  {
    echo "## NRSS Local Test Report"
    echo ""
    echo "- Timestamp (UTC): $TIMESTAMP_UTC"
    echo "- Branch: $git_branch"
    echo "- Commit: $git_sha"
    echo "- Conda env: $ENV_NAME"
    echo "- Report dir: $REPORT_DIR"
    echo "- Skip defaults: $SKIP_DEFAULTS"
    echo "- Custom repeat count: $CUSTOM_REPEAT_COUNT"
    echo "- CyRSoXS CLI override: ${CYRSOXS_CLI_DIR:-"(env default)"}"
    echo "- CyRSoXS pybind override: ${CYRSOXS_PYBIND_DIR:-"(env default)"}"
    echo "- NRSS backend: ${NRSS_BACKEND_NAME:-"(default resolution)"}"
    echo "- Steps passed: $pass_count/$total"
    echo "- Steps failed: $fail_count/$total"
    echo ""
    echo "| Step | Status | Duration (s) | Result |"
    echo "|---|---|---:|---|"
    awk -F'\t' '{printf "| %s. %s | %s | %s | %s |\n", $1, $2, $3, $4, $7}' "$STEPS_TSV"
    echo ""
    echo "### Log Files"
    awk -F'\t' '{printf "- %s. %s: `%s`\n", $1, $2, $5}' "$STEPS_TSV"
    echo "- NRSS resolution: \`$NRSS_RESOLUTION_TSV\`"
    echo "- CyRSoXS resolution: \`$CYRSOXS_RESOLUTION_TSV\`"
    echo "- Validation reference manifest: \`$VALIDATION_REFERENCE_MANIFEST_TSV\`"
  } > "$SUMMARY_MD"

  if [[ -s "$NRSS_RESOLUTION_TSV" ]]; then
    python - "$NRSS_RESOLUTION_TSV" >> "$SUMMARY_MD" <<'PY'
import csv
import pathlib
import sys

resolution_path = pathlib.Path(sys.argv[1])

print("")
print("### NRSS Resolution")
print("| Module | Resolved Path | Version | Source | Status |")
print("|---|---|---|---|---|")

with resolution_path.open(encoding="utf-8", errors="replace") as f:
    reader = csv.DictReader(f, delimiter="\t")
    for row in reader:
        if not row:
            continue
        resolved_name = (row.get("resolved_name") or "").strip() or "NRSS"
        resolved_path = (row.get("resolved_path") or "").strip() or "-"
        version = (row.get("version") or "").strip() or "-"
        git_branch = (row.get("git_branch") or "").strip()
        git_sha = (row.get("git_sha") or "").strip()
        status = (row.get("status") or "").strip() or "-"
        source = "-"
        if git_sha:
            source = f"{git_branch}@{git_sha[:7]}" if git_branch else git_sha[:7]

        def esc(value: str) -> str:
            return value.replace("|", "\\|")

        print(
            f"| `{esc(resolved_name)}` | `{esc(resolved_path)}` | `{esc(version)}` | "
            f"`{esc(source)}` | `{esc(status)}` |"
        )
PY
  fi

  if [[ -s "$CYRSOXS_RESOLUTION_TSV" ]]; then
    python - "$CYRSOXS_RESOLUTION_TSV" "$CYRSOXS_CLI_DIR" "$CYRSOXS_PYBIND_DIR" >> "$SUMMARY_MD" <<'PY'
import csv
import pathlib
import sys

resolution_path = pathlib.Path(sys.argv[1])
cli_override = sys.argv[2] or "(env default)"
pybind_override = sys.argv[3] or "(env default)"

print("")
print("### CyRSoXS Resolution")
print("| Component | Requested Override | Resolved Name | Resolved Path | Version | Source | Status |")
print("|---|---|---|---|---|---|---|")

with resolution_path.open(encoding="utf-8", errors="replace") as f:
    reader = csv.DictReader(f, delimiter="\t")
    for row in reader:
        if not row:
            continue
        component = (row.get("component") or "").strip()
        if not component:
            continue
        requested = cli_override if component == "cli" else pybind_override
        resolved_name = (row.get("resolved_name") or "").strip() or "-"
        resolved_path = (row.get("resolved_path") or "").strip() or "-"
        version = (row.get("version") or "").strip() or "-"
        git_branch = (row.get("git_branch") or "").strip()
        git_sha = (row.get("git_sha") or "").strip()
        status = (row.get("status") or "").strip() or "-"
        source = "-"
        if git_sha:
            source = f"{git_branch}@{git_sha[:7]}" if git_branch else git_sha[:7]

        def esc(value: str) -> str:
            return value.replace("|", "\\|")

        print(
            f"| `{esc(component)}` | `{esc(requested)}` | `{esc(resolved_name)}` | "
            f"`{esc(resolved_path)}` | `{esc(version)}` | `{esc(source)}` | `{esc(status)}` |"
        )
PY
  fi

  if [[ -s "$VALIDATION_REFERENCE_MANIFEST_TSV" ]]; then
    python - "$VALIDATION_REFERENCE_MANIFEST_TSV" >> "$SUMMARY_MD" <<'PY'
import csv
import pathlib
import sys

manifest_path = pathlib.Path(sys.argv[1])
with manifest_path.open(encoding="utf-8", errors="replace") as f:
    reader = csv.DictReader(f, delimiter="\t")
    rows = [row for row in reader if row]

print("")
print("### Validation Reference Provenance")
print(f"- Manifest: `{manifest_path}`")
print(f"- Hashed reference files: {len(rows)}")
PY
  fi

  if [[ -s "$SMOKE_CATALOG_TSV" ]]; then
    python - "$SMOKE_CATALOG_TSV" "$STEPS_TSV" >> "$SUMMARY_MD" <<'PY'
import pathlib
import sys

catalog_path = pathlib.Path(sys.argv[1])
steps_path = pathlib.Path(sys.argv[2])

def load_cases(path):
    statuses = {}
    if not path or not path.exists():
        return statuses
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        parts = line.split("\t")
        if len(parts) >= 2:
            statuses[parts[0]] = parts[1]
    return statuses

def soft_break_code(text):
    escaped = (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
    )
    return escaped.replace("/", "/<wbr>").replace("_", "_<wbr>")

def format_markers(markers):
    marker_list = [marker.strip() for marker in markers.split(",") if marker.strip()]
    if not marker_list:
        return "-"
    return ", ".join(f"<code>{soft_break_code(marker)}</code>" for marker in marker_list)

cpu_cases = {}
gpu_cases = {}
for line in steps_path.read_text(encoding="utf-8", errors="replace").splitlines():
    parts = line.split("\t")
    if len(parts) < 8:
        continue
    step_name = parts[1]
    case_file = pathlib.Path(parts[7])
    if step_name == "Smoke Tests (CPU Fast)":
        cpu_cases = load_cases(case_file)
    elif step_name == "Smoke Tests (GPU)":
        gpu_cases = load_cases(case_file)

print("")
print("### Smoke Tests")
print("")
for row in catalog_path.read_text(encoding="utf-8", errors="replace").splitlines():
    if not row.strip():
        continue
    parts = row.split("\t")
    name = parts[0]
    markers = parts[1] if len(parts) > 1 and parts[1] else "-"
    summary = parts[2] if len(parts) > 2 and parts[2] else "-"
    cpu = cpu_cases.get(name, "DESELECTED")
    gpu = gpu_cases.get(name, "DESELECTED")
    print(f"- <code>{soft_break_code(name)}</code>")
    print(f"  Status: CPU <code>{cpu}</code>; GPU <code>{gpu}</code>")
    print(f"  Markers: {format_markers(markers)}")
    print(f"  Summary: {summary}")
    print("")
PY
  fi

  if [[ -s "$PHYSICS_CATALOG_TSV" ]]; then
    python - "$PHYSICS_CATALOG_TSV" "$STEPS_TSV" >> "$SUMMARY_MD" <<'PY'
import pathlib
import sys

catalog_path = pathlib.Path(sys.argv[1])
steps_path = pathlib.Path(sys.argv[2])

def load_cases(path):
    statuses = {}
    if not path or not path.exists():
        return statuses
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        parts = line.split("\t")
        if len(parts) >= 2:
            statuses[parts[0]] = parts[1]
    return statuses

def soft_break_code(text):
    escaped = (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
    )
    return escaped.replace("/", "/<wbr>").replace("_", "_<wbr>")

def format_markers(markers):
    marker_list = [marker.strip() for marker in markers.split(",") if marker.strip()]
    if not marker_list:
        return "-"
    return ", ".join(f"<code>{soft_break_code(marker)}</code>" for marker in marker_list)

physics_cases = {}
for line in steps_path.read_text(encoding="utf-8", errors="replace").splitlines():
    parts = line.split("\t")
    if len(parts) < 8:
        continue
    step_name = parts[1]
    step_cmd = parts[5] if len(parts) > 5 else ""
    if step_name == "Physics Validation Tests" or (
        "tests/validation" in step_cmd and "physics_validation" in step_cmd
    ):
        physics_cases = load_cases(pathlib.Path(parts[7]))
        break

print("")
print("### Physics Tests")
print("")
for row in catalog_path.read_text(encoding="utf-8", errors="replace").splitlines():
    if not row.strip():
        continue
    parts = row.split("\t")
    test_file = parts[0]
    name = parts[1]
    markers = parts[2] if len(parts) > 2 and parts[2] else "-"
    summary = parts[3] if len(parts) > 3 and parts[3] else "-"
    status = physics_cases.get(name, "DESELECTED")
    print(f"- <code>{soft_break_code(name)}</code>")
    print(f"  Status: <code>{status}</code>")
    print(f"  File: <code>{soft_break_code(test_file)}</code>")
    print(f"  Markers: {format_markers(markers)}")
    print(f"  Description: {summary}")
    print("")
PY
  fi
}

write_metadata
capture_nrss_resolution
capture_cyrsoxs_resolution
build_validation_reference_manifest
build_smoke_catalog
build_physics_catalog

log "NRSS test report run started: $TIMESTAMP_UTC"
log "Repository: $REPO_ROOT"
log "Conda env: $ENV_NAME"
log "Report directory: $REPORT_DIR"

STEP=1
ANY_FAIL=0

ENV_SNAPSHOT_CMD=$'python - <<\'PY\'\nimport importlib\nimport platform\nimport sys\n\nmodules = [\n    "NRSS", "pytest", "numpy", "scipy", "pandas", "h5py", "xarray", "cupy", "CyRSoXS"\n]\n\nprint("python:", sys.version.replace("\\n", " "))\nprint("platform:", platform.platform())\nfor name in modules:\n    try:\n        mod = importlib.import_module(name)\n        ver = getattr(mod, "__version__", "unknown")\n        print(f"{name}: {ver}")\n    except Exception as exc:\n        print(f"{name}: NOT_AVAILABLE ({exc.__class__.__name__})")\nPY'

if [[ $SKIP_DEFAULTS -eq 0 ]]; then
  run_conda_step "Environment Snapshot" "$ENV_SNAPSHOT_CMD" "$STEP" || ANY_FAIL=1
  if [[ $ANY_FAIL -ne 0 && $STOP_ON_FAIL -eq 1 ]]; then
    generate_summary
    cat "$SUMMARY_MD"
    exit 1
  fi
  STEP=$((STEP + 1))

  run_conda_step "Smoke Tests (CPU Fast)" "python -m pytest tests/smoke -m 'not gpu' -v" "$STEP" || ANY_FAIL=1
  if [[ $ANY_FAIL -ne 0 && $STOP_ON_FAIL -eq 1 ]]; then
    generate_summary
    cat "$SUMMARY_MD"
    exit 1
  fi
  STEP=$((STEP + 1))

  run_conda_step "Smoke Tests (GPU)" "python -m pytest tests/smoke -m gpu -v" "$STEP" || ANY_FAIL=1
  if [[ $ANY_FAIL -ne 0 && $STOP_ON_FAIL -eq 1 ]]; then
    generate_summary
    cat "$SUMMARY_MD"
    exit 1
  fi
  STEP=$((STEP + 1))

  # Marker-based discovery keeps newly added physics_validation modules,
  # including experimental-reference cases such as CoreShell, in the standard lane.
  PHYSICS_CMD="python scripts/run_physics_validation_suite.py --repo-root \"$REPO_ROOT\" --catalog \"$PHYSICS_CATALOG_TSV\" --harvest-root \"$GRAPHICAL_ABSTRACTS_DIR\" --zip-path \"$GRAPHICAL_ABSTRACTS_ZIP\""
  if [[ $WRITE_PHYSICS_PLOTS -eq 0 ]]; then
    PHYSICS_CMD="$PHYSICS_CMD --no-plots"
  fi
  run_conda_step "Physics Validation Tests" "$PHYSICS_CMD" "$STEP" || ANY_FAIL=1
  if [[ $ANY_FAIL -ne 0 && $STOP_ON_FAIL -eq 1 ]]; then
    generate_summary
    cat "$SUMMARY_MD"
    exit 1
  fi
  STEP=$((STEP + 1))
fi

STEP_NAME_INDEX=1
for cmd in "${TEST_CMDS[@]}"; do
  repeat_index=1
  while [[ $repeat_index -le $CUSTOM_REPEAT_COUNT ]]; do
    step_name="Test Command ${STEP_NAME_INDEX}"
    if [[ $CUSTOM_REPEAT_COUNT -gt 1 ]]; then
      step_name="${step_name} (run ${repeat_index}/${CUSTOM_REPEAT_COUNT})"
    fi
    run_conda_step "$step_name" "$cmd" "$STEP" || ANY_FAIL=1
    if [[ $ANY_FAIL -ne 0 && $STOP_ON_FAIL -eq 1 ]]; then
      break 2
    fi
    repeat_index=$((repeat_index + 1))
    STEP=$((STEP + 1))
  done
  STEP_NAME_INDEX=$((STEP_NAME_INDEX + 1))
done

generate_summary

log ""
log "Copy/paste this summary into your PR:"
cat "$SUMMARY_MD"

if [[ -f "$GRAPHICAL_ABSTRACTS_ZIP" ]]; then
  log "Graphical abstracts zip: $GRAPHICAL_ABSTRACTS_ZIP"
fi

if [[ $ANY_FAIL -ne 0 ]]; then
  log ""
  log "Overall result: FAIL"
  exit 1
fi

log ""
log "Overall result: PASS"
exit 0
