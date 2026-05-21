#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

NPROC_PER_NODE="${NPROC_PER_NODE:-2}"
OPTIMIZER="${OPTIMIZER:-muon}"
STEPS="${STEPS:-4}"
DEVICE="${DEVICE:-auto}"
BACKEND="${BACKEND:-auto}"
BALANCE="${BALANCE:-global}"
BALANCE_COST="${BALANCE_COST:-numel}"
FUSED_COMM="${FUSED_COMM:-1}"
ATOL="${ATOL:-2e-3}"
RTOL="${RTOL:-2e-3}"
VOCAB_SIZE="${VOCAB_SIZE:-128}"
SEQ_LEN="${SEQ_LEN:-16}"
BATCH_SIZE="${BATCH_SIZE:-4}"
HIDDEN_SIZE="${HIDDEN_SIZE:-32}"
NUM_LAYERS="${NUM_LAYERS:-2}"
NUM_HEADS="${NUM_HEADS:-4}"
FFN_HIDDEN_SIZE="${FFN_HIDDEN_SIZE:-64}"
LR="${LR:-1e-3}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.0}"
SEED="${SEED:-1234}"
SOAP_PRECONDITION_FREQUENCY="${SOAP_PRECONDITION_FREQUENCY:-2}"
SOAP_MAX_PRECOND_DIM="${SOAP_MAX_PRECOND_DIM:-256}"

cmd=(
  torchrun
  --standalone
  --nproc-per-node="${NPROC_PER_NODE}"
  "${SCRIPT_DIR}/example.py"
  --optimizer "${OPTIMIZER}"
  --steps "${STEPS}"
  --device "${DEVICE}"
  --backend "${BACKEND}"
  --balance "${BALANCE}"
  --balance-cost "${BALANCE_COST}"
  --atol "${ATOL}"
  --rtol "${RTOL}"
  --vocab-size "${VOCAB_SIZE}"
  --seq-len "${SEQ_LEN}"
  --batch-size "${BATCH_SIZE}"
  --hidden-size "${HIDDEN_SIZE}"
  --num-layers "${NUM_LAYERS}"
  --num-heads "${NUM_HEADS}"
  --ffn-hidden-size "${FFN_HIDDEN_SIZE}"
  --lr "${LR}"
  --weight-decay "${WEIGHT_DECAY}"
  --seed "${SEED}"
  --soap-precondition-frequency "${SOAP_PRECONDITION_FREQUENCY}"
  --soap-max-precond-dim "${SOAP_MAX_PRECOND_DIM}"
)

if [[ "${FUSED_COMM}" == "0" ]]; then
  cmd+=(--no-fused-comm)
fi

cd "${REPO_ROOT}"
echo "[FSDP-Canzona example] ${cmd[*]}"
"${cmd[@]}"
