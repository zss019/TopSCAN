# Examples

This directory shows a few minimal configuration snippets for common use
cases. Refer to the repository root `README.md` for the full documentation.

---

## 1. Minimal Python API Usage

Script `minimal_api_demo.py` (not executable; copy it into your own project):

```python
import os, sys, torch
sys.path.insert(0, os.path.abspath("../src"))

import dgl
from topscan.models.topscan_model import TopSCAN
from topscan.modules.cross_modal_attention import CrossModalGraphAttention

N, D_text, D_img, d, C = 500, 768, 2048, 128, 5

graph = dgl.rand_graph(N, 2500)
graph = dgl.to_bidirected(graph)

text_feat = torch.randn(N, D_text)
image_feat = torch.randn(N, D_img)

model = TopSCAN(
    gnn_type="GraphSAGE",
    text_input_dim=D_text,
    image_input_dim=D_img,
    hidden_dim=d,
    num_classes=C,
    num_layers=2,
    dropout=0.5,
    cma_num_heads=4,
    cma_head_dim=32,
    warmup_fusion="concat_absdiff",
)

cross_graph = graph
out = model(graph, cross_graph, text_feat, image_feat)
print(f"output logits shape: {out.shape}")   # [N, C]
```

---

## 2. CLI Examples

### 2.1. Run on your own dataset with GraphSAGE backbone

```bash
export PYTHONPATH="../src:$PYTHONPATH"

python -m topscan.runner \
  --data_name MyDataset \
  --graph_path ./my_dataset/graph.dgl \
  --text_feature ./my_dataset/text_feat.pt \
  --visual_feature ./my_dataset/visual_feat.pt \
  --output_file ./my_dataset_results.csv \
  --gnn_type GraphSAGE --n_layers 2 --n_hidden 128 \
  --lr 0.005 --dropout 0.5 \
  --warmup_fusion concat_absdiff \
  --cma_num_heads 4 --cma_head_dim 32 --cma_norm_by dst \
  --policy_mode on --policy_k 2 --policy_q 0.95 --policy_top_p 0.15 \
  --enhance_cross_graph 1 --use_enhanced_graph_in_backbone 0 \
  --n_epochs 1000 --n_runs 5 --early_stop_patience 100 --gpu 0
```

### 2.2. Quickly try TopSCAN off (ablation baseline, i.e. plain GraphSAGE with modal concat only)

```bash
python -m topscan.runner \
  --data_name Movies --graph_path ./data/Movies/graph.dgl \
  --text_feature ./data/Movies/text_feat.pt --visual_feature ./data/Movies/visual_feat.pt \
  --output_file ./ablation_no_topscan.csv \
  --gnn_type GraphSAGE --n_layers 2 --n_hidden 128 \
  --warmup_fusion concat \
  --policy_mode off --enhance_cross_graph 0 \
  --n_epochs 500 --n_runs 3 --gpu 0
```

---

## 3. Data Format Checklist

Before running the above commands, make sure:

- [ ] `graph.dgl` is produced by `dgl.save_graphs()` and contains `graph.ndata["label"]` of shape `[N]`.
- [ ] Graph is **bidirectional** (call `dgl.to_bidirected(g)` before saving if unsure).
- [ ] `text_feat.pt` and `visual_feat.pt` are `torch.float32` tensors of shape `[N, D_text]` and `[N, D_image]` respectively.
- [ ] Node IDs in the graph are contiguous `0..N-1` and match the row order of `text_feat` / `visual_feat`.
