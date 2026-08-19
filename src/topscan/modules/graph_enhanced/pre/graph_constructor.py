import math
from typing import Dict
import torch
import dgl
from topscan.modules.graph_enhanced.framework.contracts import EnhancedGraphPack


class EnhancedGraphConstructor:
    def __init__(self, cfg):
        self.cfg = cfg

    def _topp_mask(self, edge_prob: torch.Tensor, top_p: float) -> torch.Tensor:
        n = edge_prob.numel()
        if n == 0:
            return torch.zeros(0, dtype=torch.bool, device=edge_prob.device)
        p = min(max(float(top_p), 0.0), 1.0)
        if p <= 0.0:
            return torch.zeros(n, dtype=torch.bool, device=edge_prob.device)
        k = max(1, int(math.ceil(float(n) * p)))
        k = min(k, n)
        keep_idx = torch.topk(edge_prob, k=k, largest=True, sorted=False).indices
        selected = torch.zeros(n, dtype=torch.bool, device=edge_prob.device)
        selected[keep_idx] = True
        return selected

    def forward(
        self,
        base_graph: dgl.DGLGraph,
        cand_src: torch.Tensor,
        cand_dst: torch.Tensor,
        edge_prob: torch.Tensor,
        sparse_loss: torch.Tensor,
        mode: str
    ) -> EnhancedGraphPack:
        if mode == "off" or cand_src.numel() == 0:
            stats = {
                "policy_candidate_edges": float(cand_src.numel()),
                "policy_selected_edges": 0.0,
                "policy_selected_ratio": 0.0
            }
            edge_meta = {
                "cand_src": cand_src,
                "cand_dst": cand_dst,
                "edge_prob": edge_prob
            }
            return EnhancedGraphPack(
                graph_for_backbone=base_graph,
                edge_weight=None,
                edge_meta=edge_meta,
                reg_loss=torch.zeros((), device=edge_prob.device if edge_prob.numel() > 0 else base_graph.device),
                stats=stats
            )
        if self.cfg.hard_select:
            selected = self._topp_mask(edge_prob, getattr(self.cfg, "policy_top_p", 0.01))
            sel_src = cand_src[selected]
            sel_dst = cand_dst[selected]
            sel_w = edge_prob[selected]
        else:
            selected = edge_prob > 0
            sel_src = cand_src
            sel_dst = cand_dst
            sel_w = edge_prob
        if sel_w.numel() == 0:
            stats = {
                "policy_candidate_edges": float(cand_src.numel()),
                "policy_selected_edges": 0.0,
                "policy_selected_ratio": 0.0
            }
            edge_meta = {
                "cand_src": cand_src,
                "cand_dst": cand_dst,
                "edge_prob": edge_prob,
                "selected_mask": selected
            }
            return EnhancedGraphPack(
                graph_for_backbone=base_graph,
                edge_weight=None,
                edge_meta=edge_meta,
                reg_loss=sparse_loss,
                stats=stats
            )
        base_src, base_dst = base_graph.edges()
        new_src = torch.cat([base_src, sel_src], dim=0)
        new_dst = torch.cat([base_dst, sel_dst], dim=0)
        graph_for_backbone = dgl.graph((new_src, new_dst), num_nodes=base_graph.num_nodes(), device=base_graph.device)
        edge_weight = torch.cat([torch.ones(base_src.numel(), device=sel_w.device), sel_w], dim=0)
        stats = {
            "policy_candidate_edges": float(cand_src.numel()),
            "policy_selected_edges": float(sel_src.numel()),
            "policy_selected_ratio": float(sel_src.numel()) / float(max(1, cand_src.numel()))
        }
        edge_meta = {
            "cand_src": cand_src,
            "cand_dst": cand_dst,
            "edge_prob": edge_prob,
            "selected_mask": selected
        }
        return EnhancedGraphPack(
            graph_for_backbone=graph_for_backbone,
            edge_weight=edge_weight,
            edge_meta=edge_meta,
            reg_loss=sparse_loss,
            stats=stats
        )
