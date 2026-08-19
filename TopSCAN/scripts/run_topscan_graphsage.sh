#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="${ROOT}/src:${PYTHONPATH:-}"

DATA_NAME="${1:-Movies}"
GRAPH_PATH="${2:-./data/${DATA_NAME}/graph.dgl}"
TEXT_FEAT="${3:-./data/${DATA_NAME}/text_feat.pt}"
VISUAL_FEAT="${4:-./data/${DATA_NAME}/visual_feat.pt}"
OUTPUT_FILE="${5:-./outputs/${DATA_NAME}_graphsage_topscan.csv}"
GPU="${6:-0}"

mkdir -p "$(dirname "${OUTPUT_FILE}")"

python -m topscan.runner \
  --gpu "${GPU}" \
  --data_name "${DATA_NAME}" \
  --graph_path "${GRAPH_PATH}" \
  --text_feature "${TEXT_FEAT}" \
  --visual_feature "${VISUAL_FEAT}" \
  --output_file "${OUTPUT_FILE}" \
  --gnn_type GraphSAGE \
  --n_layers 2 \
  --n_hidden 128 \
  --lr 0.005 \
  --dropout 0.5 \
  --n_epochs 1000 \
  --n_runs 5 \
  --seed 42 \
  --early_stop_patience 100 \
  --warmup_fusion concat_absdiff \
  --cma_num_heads 4 \
  --cma_head_dim 32 \
  --cma_dropout 0.2 \
  --cma_attn_dropout 0.2 \
  --cma_norm_by dst \
  --policy_mode on \
  --policy_select_mode top_fraction \
  --policy_cross_score geometric \
  --policy_k 2 \
  --policy_q 0.95 \
  --policy_top_m 2 \
  --policy_top_p 0.15 \
  --enhance_cross_graph 1 \
  --use_enhanced_graph_in_backbone 0 \
  --label_smoothing 0.1 \
  --average macro
