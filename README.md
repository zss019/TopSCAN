# TopSCAN: Structure-Aware Sparse Cross-Modal Alignment for Multimodal Attributed Graphs

Official PyTorch/DGL implementation of **TopSCAN**, a lightweight, structure-aware sparse cross-modal alignment framework for Multimodal Attributed Graphs (MAGs). TopSCAN serves as a drop-in frontend enhancement for classic GNN backbones (GraphSAGE, RevGAT, GCN, GAT, SGC) with negligible extra multimodal modeling overhead.

> **Paper Reference**: *TopSCAN: Structure-Aware Sparse Cross-Modal Alignment for Multimodal Attributed Graphs on Digital Platforms.* Submitted for review.

---

## Environment Setup

### 1. Hardware & OS

- Linux (Ubuntu 20.04+ recommended)
- NVIDIA GPU (≥ 12 GB VRAM for large datasets; CUDA 11.7 / 11.8 tested)

### 2. Create Conda Environment

```bash
conda create -n topscan python=3.10 -y
conda activate topscan
```

### 3. Install PyTorch & DGL

Please match your CUDA version. Example for **CUDA 11.8**:

```bash
# PyTorch
pip install torch==2.0.1+cu118 --index-url https://download.pytorch.org/whl/cu118

# DGL (Deep Graph Library)
pip install dgl -f https://data.dgl.ai/wheels/cu118/repo.html
```

For **CUDA 11.7**, replace `cu118` with `cu117` accordingly. For CPU-only builds, drop the CUDA suffixes.

### 4. Install Remaining Dependencies

```bash
pip install -r requirements.txt
```

`requirements.txt` contains minimal pure-Python dependencies (numpy, pandas, scikit-learn, tqdm).

### 5. (Optional) OGB Support

Only needed if you want to experiment with `ogbn-arxiv`. Install via:

```bash
pip install ogb
```

---

## Data Preparation

TopSCAN expects **three** input artifacts per dataset, placed under any directory (default layout `./data/<DATA_NAME>/`):

```
data/
└── Movies/
    ├── graph.dgl          # DGL graph object with node 'label' in ndata
    ├── text_feat.pt       # torch.Tensor of shape [N, D_text]
    └── visual_feat.pt     # torch.Tensor of shape [N, D_image]
```

### 1. DGL Graph (`graph.dgl`)

Saved with `dgl.save_graphs()`. The graph object must satisfy:

```python
graph.num_nodes() == N                          # N node count
graph.ndata["label"]  # torch.LongTensor [N]    # node class labels
```

- If you have an undirected graph, ensure it is bidirectional (use `dgl.to_bidirected()` before saving) so that GraphSAGE / RevGAT operate correctly.
- Edge features are optional and not required by the current implementation.

### 2. Node Features (`text_feat.pt`, `visual_feat.pt`)

- `text_feat.pt` — Pre-extracted textual node embeddings (e.g. from BERT / Sentence-BERT / CLIP-Text). Shape `[N, D_text]`, `torch.float32`.
- `visual_feat.pt` — Pre-extracted visual node embeddings (e.g. from ResNet / ViT / CLIP-Vision). Shape `[N, D_image]`, `torch.float32`.

Both tensors should be row-wise aligned with node IDs in `graph.dgl` (the i-th row corresponds to node i).

### 3. Train / Val / Test Splits

By default, `topscan.data.graph_data.load_data` randomly splits nodes with ratios `train:val:test = 60%:20%:20%` (seed fixed to 42). To reproduce exact paper numbers, you can:

- Either use the defaults (for a quick comparison in the same split regime), or
- Pre-generate split indices and pass them by modifying the `load_data` helper.

---

## Quick Start

### 1. Clone & Enter Repository

```bash
git clone <your-repo-url> TopSCAN && cd TopSCAN
```

### 2. Verify Installation & Imports

```bash
export PYTHONPATH="$(pwd)/src:$PYTHONPATH"
python -c "from topscan.models.topscan_model import TopSCAN; print('TopSCAN import OK')"
```

### 3. Single-Run — GraphSAGE + TopSCAN

```bash
# Place your Movies data under ./data/Movies/ as described above, then run:
bash scripts/run_topscan_graphsage.sh Movies
```

The script accepts positional arguments:

```bash
bash scripts/run_topscan_graphsage.sh <DATA_NAME> <GRAPH_PATH> <TEXT_FEAT> <VISUAL_FEAT> <OUTPUT_CSV> <GPU_ID>
```

### 4. Single-Run — RevGAT + TopSCAN

```bash
bash scripts/run_topscan_revgat.sh Toys
```

### 5. Grid Search (Hyper-Parameter Sweep)

```bash
bash scripts/run_grid_search.sh Grocery
```

This sweeps over backbone (GraphSAGE / RevGAT), hidden dims, learning rates, dropout rates, CMA heads, pruning policy thresholds, etc., and writes every configuration's mean±std metrics to a CSV.

---

## Reproducing Main Results

The following configurations correspond to the main experiments in our paper (Overall Performance Comparison on Movies / Toys / Grocery / Reddit-S).

### GraphSAGE + TopSCAN

```bash
bash scripts/run_topscan_graphsage.sh <DATASET_NAME>
```

Key configuration (already set inside script):

- `--gnn_type GraphSAGE --n_layers 2 --n_hidden 128 --lr 0.005 --dropout 0.5`
- `--warmup_fusion concat_absdiff --cma_num_heads 4 --cma_head_dim 32 --cma_norm_by dst`
- `--policy_mode on --policy_k 2 --policy_q 0.95 --policy_top_p 0.15`
- `--enhance_cross_graph 1 --use_enhanced_graph_in_backbone 0`
- `--n_runs 5 --n_epochs 1000 --early_stop_patience 100`

### RevGAT + TopSCAN

```bash
bash scripts/run_topscan_revgat.sh <DATASET_NAME>
```

Key configuration:

- `--gnn_type RevGAT --n_layers 3 --n_hidden 256 --n_heads 3 --lr 0.005 --dropout 0.5`
- `--attn_drop 0.1 --edge_drop 0.1 --alpha 0.1 --use_attn_dst 1 --use_symmetric_norm 1`
- Same TopSCAN / Policy hyper-parameters as GraphSAGE variant (CMA heads increased to 8 for RevGAT's larger capacity).

For reviewers' convenience, the paper's reported numbers use the same `n_runs=5`, `seed=42..46` scheme as the defaults here.

---

## Directory Structure

```
TopSCAN/
├── scripts/
│   ├── run_topscan_graphsage.sh   # GraphSAGE+TopSCAN quick run
│   ├── run_topscan_revgat.sh      # RevGAT+TopSCAN quick run
│   └── run_grid_search.sh         # Full grid search over hyper-params
├── src/
│   └── topscan/
│       ├── __init__.py
│       ├── runner.py              # **Main entry point** (CLI + train/eval loop)
│       ├── data/
│       │   └── graph_data.py      # DGL graph loading + split + seed
│       ├── models/
│       │   ├── topscan_model.py   # TopSCAN class (project + CMA + fuse + GNN + cls)
│       │   └── backbones/
│       │       ├── gcn.py         # Minimal GCN
│       │       ├── gat.py         # Minimal GAT
│       │       ├── graphsage.py   # Minimal GraphSAGE
│       │       └── RevGAT/        # RevGAT full implementation
│       ├── modules/
│       │   ├── cross_modal_attention.py        # CrossModalGraphAttention (CMA)
│       │   └── graph_enhanced/
│       │       ├── framework/contracts.py      # Typing / data-pack contracts
│       │       └── pre/
│       │           ├── candidate_builder.py    # k-hop + source-wise quantile + top-m/p
│       │           └── graph_constructor.py    # Enhanced graph construction orchestrator
│       └── utils/
│           ├── loss_function.py   # CE with label smoothing, accuracy/F1, EarlyStopping, LR warmup
│           └── model_config.py    # argparse groups: add_common_args / add_topscan_args
├── examples/                      # (Optional) sample configs / tutorials
├── requirements.txt
├── .gitignore
├── LICENSE
└── README.md
```

---

## Citation


---

## License

This repository is released under the [MIT License](LICENSE). The RevGAT sub-module (see `src/topscan/models/backbones/RevGAT/`) retains its original license (included in-situ where applicable).

---

## Contact

For code-related issues, please open an issue in the repository. For paper correspondence, please contact the authors via the review process channel.
