#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  ./scripts/validate-submission.sh <ping_url> [repo_dir]

Examples:
  ./scripts/validate-submission.sh https://sriman676-soc-agentlab.hf.space
  ./scripts/validate-submission.sh https://huggingface.co/spaces/Sriman676/soc-agentlab .
EOF
}

if [[ $# -lt 1 || $# -gt 2 ]]; then
  usage
  exit 2
fi

RAW_URL="$1"
REPO_DIR="${2:-.}"

normalize_url() {
  local input="$1"
  if [[ "$input" =~ ^https://huggingface\.co/spaces/([^/]+)/([^/]+)$ ]]; then
    local owner="${BASH_REMATCH[1]}"
    local space="${BASH_REMATCH[2]}"
    echo "https://${owner,,}-${space,,}.hf.space"
    return
  fi
  echo "$input"
}

PING_URL="$(normalize_url "$RAW_URL")"

require_200() {
  local method="$1"
  local url="$2"
  local body="${3:-}"
  local code

  if [[ -n "$body" ]]; then
    code="$(curl -sS -o /tmp/validate_resp.json -w '%{http_code}' -X "$method" "$url" -H 'Content-Type: application/json' -d "$body")"
  else
    code="$(curl -sS -o /tmp/validate_resp.json -w '%{http_code}' -X "$method" "$url")"
  fi

  if [[ "$code" != "200" ]]; then
    echo "[FAIL] $method $url -> HTTP $code"
    echo "Response:"
    cat /tmp/validate_resp.json || true
    exit 1
  fi

  echo "[PASS] $method $url -> HTTP 200"
}

echo "Using Space URL: $PING_URL"

require_200 GET "$PING_URL/"
if ! grep -qi 'status' /tmp/validate_resp.json; then
  echo "[WARN] Root response does not include 'status' key."
fi

require_200 GET "$PING_URL/reset"
if ! grep -qi 'observation' /tmp/validate_resp.json; then
  echo "[WARN] GET /reset response does not include 'observation' key."
fi

require_200 POST "$PING_URL/reset" '{}'
if ! grep -qi 'observation' /tmp/validate_resp.json; then
  echo "[WARN] POST /reset response does not include 'observation' key."
fi

require_200 GET "$PING_URL/state"

pushd "$REPO_DIR" >/dev/null

echo "[INFO] Checking required files..."
for required in Dockerfile inference.py openenv.yaml tasks.py graders.py validate_env.py; do
  if [[ ! -f "$required" ]]; then
    echo "[FAIL] Missing required file: $required"
    exit 1
  fi
done
echo "[PASS] Required files found"

echo "[INFO] Running openenv validation..."
./openenv validate

echo "[INFO] Running strict task/grader checks..."
python - <<'PY'
import importlib
import yaml

from tasks import TASKS
from graders import grade_all

with open("openenv.yaml", "r", encoding="utf-8") as f:
    cfg = yaml.safe_load(f) or {}

task_items = cfg.get("tasks") or []
if not isinstance(task_items, list) or len(task_items) < 5:
  raise SystemExit("[FAIL] openenv.yaml must define at least 5 tasks")

graded = 0
for item in task_items:
    if not isinstance(item, dict):
        continue
    grader_path = item.get("grader")
    if isinstance(grader_path, str) and grader_path.strip():
        graded += 1
        mod, fn = grader_path.rsplit(".", 1)
        fn_obj = getattr(importlib.import_module(mod), fn)
        if not callable(fn_obj):
            raise SystemExit(f"[FAIL] Grader is not callable: {grader_path}")
        out = float(fn_obj([]))
        if not (0.0 < out < 1.0):
            raise SystemExit(f"[FAIL] Grader output out of range for {grader_path}: {out}")

    score = item.get("score")
    if score is not None:
        sv = float(score)
        if not (0.0 < sv < 1.0):
            raise SystemExit(f"[FAIL] Task score out of range in openenv.yaml: {sv}")

if graded < 5:
  raise SystemExit("[FAIL] Fewer than 5 tasks with graders")

if len(TASKS) < 5:
  raise SystemExit("[FAIL] TASKS must contain at least 5 entries")

callable_graders = sum(1 for cfg in TASKS.values() if callable(cfg.get("grader")))
if callable_graders < 5:
  raise SystemExit("[FAIL] Fewer than 5 callable graders in TASKS")

for name, cfg in TASKS.items():
    val = float(cfg["grader"]([]))
    if not (0.0 < val < 1.0):
        raise SystemExit(f"[FAIL] TASKS grader output out of range for {name}: {val}")

for key, value in grade_all([]).items():
    fv = float(value)
    if not (0.0 < fv < 1.0):
        raise SystemExit(f"[FAIL] grade_all output out of range for {key}: {fv}")

print("[PASS] Strict task/grader checks")
PY

echo "[INFO] Running baseline inference check..."
start_ts="$(date +%s)"
python inference.py >/tmp/precheck_inference.log
end_ts="$(date +%s)"
duration="$((end_ts - start_ts))"

if [[ "$duration" -ge 1200 ]]; then
  echo "[FAIL] inference.py exceeded 20 minute limit (${duration}s)"
  exit 1
fi

if ! head -n 1 /tmp/precheck_inference.log | grep -Eq '^\[START\] task=.* env=.* model=.*$'; then
  echo "[FAIL] Missing or malformed [START] log line"
  exit 1
fi

if ! grep -Eq '^\[STEP\] step=[0-9]+ action=[^ ]+.* reward=[0-9]+\.[0-9]{2} done=(true|false) error=.*$' /tmp/precheck_inference.log; then
  echo "[FAIL] Missing or malformed [STEP] log line(s)"
  exit 1
fi

if ! tail -n 1 /tmp/precheck_inference.log | grep -Eq '^\[END\] success=(true|false) steps=[0-9]+ score=[0-9]+\.[0-9]{3}.*$'; then
  echo "[FAIL] Missing or malformed [END] log line"
  exit 1
fi

echo "[PASS] Baseline inference check (${duration}s)"

echo "[INFO] Building Docker image..."
docker build -t soc-agentlab-precheck . >/tmp/docker_precheck.log

echo "[PASS] Docker build succeeded"

echo "[DONE] All pre-submission checks passed"
popd >/dev/null
