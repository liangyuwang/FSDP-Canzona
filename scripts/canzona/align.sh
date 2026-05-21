#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

NPROC_PER_NODE="${NPROC_PER_NODE:-2}"
DEVICE="${DEVICE:-auto}"
BACKEND="${BACKEND:-auto}"
STEPS="${STEPS:-4}"
ATOL="${ATOL:-2e-3}"
RTOL="${RTOL:-2e-3}"

run_case() {
  local name="$1"
  local optimizer="$2"
  local balance="$3"
  local balance_cost="$4"
  local fused_comm="$5"
  local overlap="$6"

  echo
  echo "========== ${name} =========="
  NPROC_PER_NODE="${NPROC_PER_NODE}" \
  DEVICE="${DEVICE}" \
  BACKEND="${BACKEND}" \
  STEPS="${STEPS}" \
  ATOL="${ATOL}" \
  RTOL="${RTOL}" \
  OPTIMIZER="${optimizer}" \
  BALANCE="${balance}" \
  BALANCE_COST="${balance_cost}" \
  FUSED_COMM="${fused_comm}" \
  OVERLAP="${overlap}" \
  bash "${SCRIPT_DIR}/example.sh"
}

run_case "muon-simple-serial" "muon" "no" "numel" "1" "none"
run_case "muon-global-serial" "muon" "global" "flops" "1" "none"
run_case "muon-global-full-overlap" "muon" "global" "flops" "1" "full"
run_case "soap-global-serial" "soap" "global" "numel" "1" "none"
run_case "soap-global-full-overlap" "soap" "global" "numel" "1" "full"
