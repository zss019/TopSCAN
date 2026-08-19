from typing import Dict
import torch
import dgl


def _cross_score(
    s_text: torch.Tensor,
    s_image: torch.Tensor,
    mode: str,
    route: str = "original",
) -> torch.Tensor:
    if mode == "geometric":
        if route == "softplus":
            st = torch.nn.functional.softplus(s_text)
            si = torch.nn.functional.softplus(s_image)
            return torch.sqrt(st * si + 1e-8)
        if route == "sign_preserving":
            sign = torch.sign(s_text) * torch.sign(s_image)
            mag = torch.sqrt(torch.abs(s_text) * torch.abs(s_image) + 1e-8)
            return sign * mag
        # default/original: keep current behavior for backward compatibility
        return torch.sqrt(torch.relu(s_text) * torch.relu(s_image))
    return 0.5 * (s_text + s_image)


def _select_by_src(u: torch.Tensor, score: torch.Tensor, q: float, top_m: int):
    order = torch.argsort(u)
    u = u[order]
    # Guard against NaN/Inf so quantile comparison does not collapse to all-False.
    score = torch.nan_to_num(score[order], nan=0.0, posinf=1.0, neginf=-1.0)
    _, counts = torch.unique_consecutive(u, return_counts=True)
    keep_mask = torch.zeros_like(score, dtype=torch.bool)
    ptr = 0
    for i in range(counts.numel()):
        cnt = int(counts[i].item())
        s = score[ptr:ptr + cnt]
        thresh = torch.quantile(s, q) if cnt > 1 else s[0]
        local_keep = s >= thresh
        if top_m > 0 and int(local_keep.sum().item()) > top_m:
            local_idx = torch.topk(s, k=top_m, largest=True).indices
            local_keep = torch.zeros_like(local_keep)
            local_keep[local_idx] = True
        keep_mask[ptr:ptr + cnt] = local_keep
        ptr += cnt
    return order, keep_mask


class CandidateBuilder:
    def __init__(self, cfg):
        self.cfg = cfg

    def build_cache(self, graph: dgl.DGLGraph, text_feat: torch.Tensor, image_feat: torch.Tensor) -> Dict[str, torch.Tensor]:
        device = text_feat.device
        src_e, dst_e = graph.edges()
        exist = set((int(s), int(d)) for s, d in zip(src_e.cpu().tolist(), dst_e.cpu().tolist()))
        khop = dgl.khop_graph(graph, k=self.cfg.policy_k)
        u, v = khop.edges()
        keep = [(int(uu), int(vv)) for uu, vv in zip(u.cpu().tolist(), v.cpu().tolist()) if uu != vv and (int(uu), int(vv)) not in exist]
        if len(keep) == 0:
            z_long = torch.zeros(0, dtype=torch.long, device=device)
            z_feat = torch.zeros(0, 5, dtype=torch.float32, device=device)
            z_float = torch.zeros(0, dtype=torch.float32, device=device)
            return {"cand_src": z_long, "cand_dst": z_long, "cand_feat": z_feat, "raw_score": z_float}
        uv = torch.tensor(keep, dtype=torch.long, device=device)
        cand_src = uv[:, 0]
        cand_dst = uv[:, 1]
        t_norm = torch.nn.functional.normalize(text_feat, p=2, dim=1)
        i_norm = torch.nn.functional.normalize(image_feat, p=2, dim=1)
        s_text = (t_norm[cand_src] * t_norm[cand_dst]).sum(dim=1)
        s_image = (i_norm[cand_src] * i_norm[cand_dst]).sum(dim=1)
        cross_route = getattr(self.cfg, "policy_cross_route", "original")
        s_cross = _cross_score(s_text, s_image, self.cfg.policy_cross_mode, cross_route)
        raw_score = self.cfg.policy_alpha * s_text + self.cfg.policy_beta * s_image + self.cfg.policy_gamma * s_cross
        raw_score = torch.nan_to_num(raw_score, nan=0.0, posinf=1.0, neginf=-1.0)
        order, keep_mask = _select_by_src(cand_src, raw_score, self.cfg.policy_q, self.cfg.policy_top_m)
        cand_src = cand_src[order][keep_mask]
        cand_dst = cand_dst[order][keep_mask]
        s_text = s_text[order][keep_mask]
        s_image = s_image[order][keep_mask]
        s_cross = s_cross[order][keep_mask]
        raw_score = raw_score[order][keep_mask]
        if cand_src.numel() == 0:
            z_long = torch.zeros(0, dtype=torch.long, device=device)
            z_feat = torch.zeros(0, 5, dtype=torch.float32, device=device)
            z_float = torch.zeros(0, dtype=torch.float32, device=device)
            return {"cand_src": z_long, "cand_dst": z_long, "cand_feat": z_feat, "raw_score": z_float}
        deg = (graph.in_degrees() + graph.out_degrees()).float()
        max_deg = torch.clamp(deg.max(), min=1.0)
        deg_gap = torch.abs(deg[cand_src] - deg[cand_dst]) / max_deg
        neigh = [set() for _ in range(graph.num_nodes())]
        for a, b in zip(src_e.cpu().tolist(), dst_e.cpu().tolist()):
            aa = int(a)
            bb = int(b)
            neigh[aa].add(bb)
            neigh[bb].add(aa)
        jaccard = torch.zeros(cand_src.numel(), dtype=torch.float32, device=device)
        for idx in range(cand_src.numel()):
            uu = int(cand_src[idx].item())
            vv = int(cand_dst[idx].item())
            nu = neigh[uu]
            nv = neigh[vv]
            inter = len(nu.intersection(nv))
            union = len(nu.union(nv))
            jaccard[idx] = float(inter) / float(union) if union > 0 else 0.0
        cand_feat = torch.stack([s_text.float(), s_image.float(), s_cross.float(), jaccard, deg_gap.float()], dim=1)
        return {"cand_src": cand_src.long(), "cand_dst": cand_dst.long(), "cand_feat": cand_feat, "raw_score": raw_score.float()}

    def forward(self, graph: dgl.DGLGraph, text_feat: torch.Tensor, image_feat: torch.Tensor) -> Dict[str, torch.Tensor]:
        return self.build_cache(graph, text_feat, image_feat)
