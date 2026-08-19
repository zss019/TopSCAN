#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="${ROOT}/src:${PYTHONPATH:-}"

DATA_NAME="${1:-Movies}"
GRAPH_PATH="${2:-./data/${DATA_NAME}/graph.dgl}"
TEXT_FEAT="${3:-./data/${DATA_NAME}/text_feat.pt}"
VISUAL_FEAT="${4:-./data/${DATA_NAME}/visual_feat.pt}"
OUTPUT_FILE="${5:-./outputs/${DATA_NAME}_gridsearch.csv}"
GPU="${6:-0}"

mkdir -p "$(dirname "${OUTPUT_FILE}")"

python -m topscan.runner \
  --gpu "${GPU}" \
  --data_name "${DATA_NAME}" \
  --graph_path "${GRAPH_PATH}" \
  --text_feature "${TEXT_FEAT}" \
  --visual_feature "${VISUAL_FEAT}" \
  --output_file "${OUTPUT_FILE}" \
  --grid_search 1 \
  --grid_gnn_types "GraphSAGE,RevGAT" \
  --grid_layers "2,3" \
  --grid_hidden_units "128,256" \
  --grid_lrs "0.005,0.01" \
  --grid_dropouts "0.3,0.5" \
  --grid_fusions "concat_absdiff" \
  --grid_cma_num_heads "4,8" \
  --grid_cma_head_dims "32" \
  --grid_cma_attn_dropouts "0.1,0.2" \
  --grid_cma_norm_bys "src,dst" \
  --grid_policy_ks "2,3" \
  --grid_policy_qs "0.90,0.95" \
  --grid_policy_top_ms "2,3" \
  --grid_policy_top_ps "0.10,0.15" \
  --policy_mode on \
  --policy_select_mode top_fraction \
  --policy_cross_score geometric \
  --enhance_cross_graph 1 \
  --use_enhanced_graph_in_backbone 0 \
  --n_epochs 1000 \
  --n_runs 5 \
  --seed 42 \
  --early_stop_patience 100 \
  --label_smoothing 0.1 \
  --average macro
