#!/usr/bin/env bash
set -u
set -o pipefail

usage() {
  cat <<'EOF'
Usage:
  scripts/run_local_test_report.sh [options]

Options:
  -e, --env NAME            Conda environment name (default: $NRSS_TEST_ENV or mar2025)
  -r, --report-root PATH    Report root directory (default: test-reports)
  --cyrsoxs-cli-dir PATH    Prepend PATH to CLI lookup inside each conda-run step.
  --cyrsoxs-pybind-dir PATH Prepend PATH to PYTHONPATH inside each conda-run step.
  --cmd "COMMAND"           Add a test command to run in conda env.
                            Can be passed multiple times.
                            If not provided, default smoke tests (CPU+GPU) are executed.
  --stop-on-fail            Stop after first failing step.
  -h, --help                Show this help.

Environment overrides:
  NRSS_TEST_CYRSOXS_CLI_DIR     Same as --cyrsoxs-cli-dir.
  NRSS_TEST_CYRSOXS_PYBIND_DIR  Same as --cyrsoxs-pybind-dir.

Examples:
  scripts/run_local_test_report.sh
  NRSS_TEST_ENV=mar2025 scripts/run_local_test_report.sh
  scripts/run_local_test_report.sh -e nrss-dev \\
    --cyrsoxs-cli-dir /path/to/cyrsoxs/build \\
    --cyrsoxs-pybind-dir /path/to/cyrsoxs/build-pybind
  NRSS_TEST_CYRSOXS_CLI_DIR=/path/to/cyrsoxs/build \\
    NRSS_TEST_CYRSOXS_PYBIND_DIR=/path/to/cyrsoxs/build-pybind \\
    scripts/run_local_test_report.sh -e nrss-dev
  scripts/run_local_test_report.sh --cmd "python -m pytest -m 'not slow' -q"
  scripts/run_local_test_report.sh --cmd "python -m pytest tests/smoke -m gpu -q"
  scripts/run_local_test_report.sh -e mar2025 --cmd "python -m pytest tests/validation -q"
EOF
}

ENV_NAME="${NRSS_TEST_ENV:-mar2025}"
REPORT_ROOT="test-reports"
STOP_ON_FAIL=0
CYRSOXS_CLI_DIR="${NRSS_TEST_CYRSOXS_CLI_DIR:-${NRSS_TEST_PATH_PREPEND:-}}"
CYRSOXS_PYBIND_DIR="${NRSS_TEST_CYRSOXS_PYBIND_DIR:-${NRSS_TEST_PYTHONPATH_PREPEND:-}}"
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
    --cyrsoxs-cli-dir)
      CYRSOXS_CLI_DIR="${2:-}"
      shift 2
      ;;
    --cyrsoxs-pybind-dir)
      CYRSOXS_PYBIND_DIR="${2:-}"
      shift 2
      ;;
    --cmd)
      TEST_CMDS+=("${2:-}")
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

: > "$RUN_LOG"
: > "$STEPS_TSV"

log() {
  local msg="$1"
  printf '%s\n' "$msg" | tee -a "$RUN_LOG"
}

slugify() {
  echo "$1" | tr '[:upper:]' '[:lower:]' | tr -cs 'a-z0-9' '_' | sed 's/^_//;s/_$//'
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
    echo "cyrsoxs_cli_dir=$CYRSOXS_CLI_DIR"
    echo "cyrsoxs_pybind_dir=$CYRSOXS_PYBIND_DIR"
    if command -v nvidia-smi >/dev/null 2>&1; then
      echo "nvidia_smi=present"
      nvidia-smi -L || true
    else
      echo "nvidia_smi=not_found"
    fi
  } > "$METADATA_TXT"
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
        summary = doc[0] if doc else ""
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

  inner_cmd="$cmd"
  if [[ -n "$CYRSOXS_PYBIND_DIR" ]]; then
    inner_cmd="export PYTHONPATH=\"${CYRSOXS_PYBIND_DIR}\${PYTHONPATH:+:\$PYTHONPATH}\""$'\n'"$inner_cmd"
  fi
  if [[ -n "$CYRSOXS_CLI_DIR" ]]; then
    inner_cmd="export PATH=\"${CYRSOXS_CLI_DIR}:\$PATH\""$'\n'"$inner_cmd"
  fi

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
    echo "- Steps passed: $pass_count/$total"
    echo "- Steps failed: $fail_count/$total"
    echo ""
    echo "| Step | Status | Duration (s) | Result |"
    echo "|---|---|---:|---|"
    awk -F'\t' '{printf "| %s. %s | %s | %s | %s |\n", $1, $2, $3, $4, $7}' "$STEPS_TSV"
    echo ""
    echo "### Log Files"
    awk -F'\t' '{printf "- %s. %s: `%s`\n", $1, $2, $5}' "$STEPS_TSV"
  } > "$SUMMARY_MD"

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
print("### Smoke Test Meanings")
print("| Test | Markers | Summary | CPU smoke | GPU smoke |")
print("|---|---|---|---|---|")
for row in catalog_path.read_text(encoding="utf-8", errors="replace").splitlines():
    if not row.strip():
        continue
    parts = row.split("\t")
    name = parts[0]
    markers = parts[1] if len(parts) > 1 and parts[1] else "-"
    summary = parts[2] if len(parts) > 2 and parts[2] else "-"
    summary = summary.replace("|", "\\|")
    cpu = cpu_cases.get(name, "DESELECTED")
    gpu = gpu_cases.get(name, "DESELECTED")
    print(f"| `{name}` | `{markers}` | {summary} | {cpu} | {gpu} |")
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

physics_cases = {}
for line in steps_path.read_text(encoding="utf-8", errors="replace").splitlines():
    parts = line.split("\t")
    if len(parts) < 8:
        continue
    if parts[1] == "Physics Validation Tests":
        physics_cases = load_cases(pathlib.Path(parts[7]))
        break

print("")
print("### Physics Validation Meanings")
print("| Test | File | Markers | Summary | Status |")
print("|---|---|---|---|---|")
for row in catalog_path.read_text(encoding="utf-8", errors="replace").splitlines():
    if not row.strip():
        continue
    parts = row.split("\t")
    test_file = parts[0]
    name = parts[1]
    markers = parts[2] if len(parts) > 2 and parts[2] else "-"
    summary = parts[3] if len(parts) > 3 and parts[3] else "-"
    summary = summary.replace("|", "\\|")
    status = physics_cases.get(name, "DESELECTED")
    print(f"| `{name}` | `{test_file}` | `{markers}` | {summary} | {status} |")
PY
  fi
}

write_metadata
build_smoke_catalog
build_physics_catalog

log "NRSS test report run started: $TIMESTAMP_UTC"
log "Repository: $REPO_ROOT"
log "Conda env: $ENV_NAME"
log "Report directory: $REPORT_DIR"

STEP=1
ANY_FAIL=0

ENV_SNAPSHOT_CMD=$'python - <<\'PY\'\nimport importlib\nimport platform\nimport sys\n\nmodules = [\n    "NRSS", "pytest", "numpy", "scipy", "pandas", "h5py", "xarray", "cupy", "CyRSoXS", "cyrsoxs"\n]\n\nprint("python:", sys.version.replace("\\n", " "))\nprint("platform:", platform.platform())\nfor name in modules:\n    try:\n        mod = importlib.import_module(name)\n        ver = getattr(mod, "__version__", "unknown")\n        print(f"{name}: {ver}")\n    except Exception as exc:\n        print(f"{name}: NOT_AVAILABLE ({exc.__class__.__name__})")\nPY'

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

run_conda_step "Physics Validation Tests" "python -m pytest tests/validation -m physics_validation -v" "$STEP" || ANY_FAIL=1
if [[ $ANY_FAIL -ne 0 && $STOP_ON_FAIL -eq 1 ]]; then
  generate_summary
  cat "$SUMMARY_MD"
  exit 1
fi
STEP=$((STEP + 1))

STEP_NAME_INDEX=1
for cmd in "${TEST_CMDS[@]}"; do
  run_conda_step "Test Command ${STEP_NAME_INDEX}" "$cmd" "$STEP" || ANY_FAIL=1
  if [[ $ANY_FAIL -ne 0 && $STOP_ON_FAIL -eq 1 ]]; then
    break
  fi
  STEP_NAME_INDEX=$((STEP_NAME_INDEX + 1))
  STEP=$((STEP + 1))
done

generate_summary

log ""
log "Copy/paste this summary into your PR:"
cat "$SUMMARY_MD"

if [[ $ANY_FAIL -ne 0 ]]; then
  log ""
  log "Overall result: FAIL"
  exit 1
fi

log ""
log "Overall result: PASS"
exit 0
