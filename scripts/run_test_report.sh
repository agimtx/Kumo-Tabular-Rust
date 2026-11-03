#!/usr/bin/env bash
# Run the medium-model test matrix used by TEST_REPORT.md.
#
# Every Rust evaluation reads its ONNX model from models/ and every official
# comparison reads its FP32 checkpoint from pt/. Results are cached in output/,
# so rerunning resumes missing reports instead of repeating finished work.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

: "${ORT_DYLIB_PATH:?set ORT_DYLIB_PATH to the ONNX Runtime shared library}"
export CARGO_HOME="${CARGO_HOME:-$ROOT/.cargo-home}"
export ORT_DYLIB_PATH

CONTEXTS=(16 32 64)
TASKS=(classification regression)
PRECISIONS=(fp16 fp8 int4)
SEED=42

cargo build --locked --offline --release --bin evaluate_full
mkdir -p output

rust_report() {
  local task=$1 precision=$2 context=$3
  if [[ "$context" == 32 ]]; then
    echo "output/medium-${task}-${precision}-full-evaluation-rust.json"
  else
    echo "output/medium-${task}-${precision}-full-context-${context}-evaluation-rust.json"
  fi
}

official_report() {
  local task=$1 precision=$2 context=$3
  local suffix=""
  [[ "$context" == 32 ]] || suffix="-context-${context}"
  echo "output/medium-${task}-${precision}${suffix}-official-comparison.json"
}

trace_report() {
  local task=$1 context=$2
  echo "output/medium-${task}-context-${context}-seed-${SEED}-trace.json"
}

for context in "${CONTEXTS[@]}"; do
  for task in "${TASKS[@]}"; do
    trace="$(trace_report "$task" "$context")"
    if [[ ! -s "$trace" ]]; then
      echo "=== $task context=$context: tracing FP16 batches ==="
      target/release/evaluate_full --seed "$SEED" --size medium --task "$task" \
        --precision fp16 --context-rows "$context" --trace
    fi
    for precision in "${PRECISIONS[@]}"; do
      report="$(rust_report "$task" "$precision" "$context")"
      if [[ -s "$report" ]]; then
        echo "=== $task $precision context=$context: already present ==="
        continue
      fi
      echo "=== $task $precision context=$context: Rust inference ==="
      target/release/evaluate_full --seed "$SEED" --size medium --task "$task" \
        --precision "$precision" --context-rows "$context"
    done
  done
done

for context in "${CONTEXTS[@]}"; do
  for task in "${TASKS[@]}"; do
    for precision in "${PRECISIONS[@]}"; do
      report="$(official_report "$task" "$precision" "$context")"
      if [[ -s "$report" ]]; then
        echo "=== $task $precision context=$context: official comparison present ==="
        continue
      fi
      echo "=== $task $precision context=$context: official PyTorch replay ==="
      .venv-official/bin/python scripts/compare_official_medium.py \
        --task "$task" --precision "$precision" --context-rows "$context" --seed "$SEED"
    done
  done
done

.venv-official/bin/python scripts/generate_test_report.py