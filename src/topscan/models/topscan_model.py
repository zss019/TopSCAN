import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn.pytorch import SGConv

from topscan.models.backbones.gcn import GCN
from topscan.models.backbones.gat import GAT
from topscan.models.backbones.graphsage import GraphSAGE
from topscan.models.backbones.RevGAT.model import RevGAT
from topscan.modules.cross_modal_attention import CrossModalGraphAttention


class TopSCAN(nn.Module):
    def __init__(
        self,
        gnn_type,
        text_input_dim,
        image_input_dim,
        hidden_dim,
        num_classes,
        num_layers,
        dropout,
        cma_num_heads=4,
        cma_head_dim=32,
        cma_dropout=0.2,
        cma_attn_dropout=0.2,
        cma_norm_by="dst",
        warmup_fusion="concat_absdiff",
        warmup_alpha=0.5,
        **kwargs,
    ):
        super().__init__()
        self.gnn_type = gnn_type
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.num_layers = num_layers
        self.dropout_ratio = dropout
        self.warmup_fusion = warmup_fusion
        self.warmup_alpha = warmup_alpha
        self._hparams = dict(kwargs)

        self.text_proj = nn.Linear(text_input_dim, hidden_dim)
        self.image_proj = nn.Linear(image_input_dim, hidden_dim)
        self.input_dropout = nn.Dropout(dropout)

        self.cma = CrossModalGraphAttention(
            hidden_dim=hidden_dim,
            num_heads=cma_num_heads,
            head_dim=cma_head_dim,
            dropout=cma_dropout,
            attn_dropout=cma_attn_dropout,
            norm_by=cma_norm_by,
        )

        fused_input_dim = self._get_fused_input_dim(hidden_dim)
        if gnn_type == "GCN":
            self.gnn = self._build_gcn(fused_input_dim, hidden_dim, num_layers)
        elif gnn_type == "GAT":
            heads = kwargs.get("heads", 3)
            self.gnn = self._build_gat(fused_input_dim, hidden_dim, num_layers, heads)
        elif gnn_type == "GraphSAGE":
            aggregator = kwargs.get("aggregator", "mean")
            self.gnn = self._build_graphsage(fused_input_dim, hidden_dim, num_layers, aggregator)
        elif gnn_type == "RevGAT":
            heads = kwargs.get("heads", 3)
            self.gnn = self._build_revgat(fused_input_dim, hidden_dim, num_layers, heads)
        elif gnn_type == "SGC":
            k = kwargs.get("k", 2)
            self.gnn = self._build_sgc(fused_input_dim, hidden_dim, k)
        else:
            raise ValueError(f"Unknown GNN type: {gnn_type}")

        self.classifier = nn.Linear(hidden_dim, num_classes)
        self.dropout = nn.Dropout(dropout)

    def _build_gcn(self, in_dim, out_dim, layers):
        return GCN(in_dim, out_dim, out_dim, layers, F.relu, self.dropout_ratio)

    def _build_gat(self, in_dim, out_dim, layers, heads):
        return GAT(
            in_dim,
            out_dim,
            out_dim,
            layers,
            heads,
            F.relu,
            self.dropout_ratio,
            0.1,
            0.1,
            True,
            True,
        )

    def _build_graphsage(self, in_dim, out_dim, layers, aggregator):
        return GraphSAGE(
            in_dim,
            out_dim,
            out_dim,
            layers,
            F.relu,
            self.dropout_ratio,
            aggregator,
        )

    def _build_revgat(self, in_dim, out_dim, layers, heads):
        use_attn_dst = bool(int(self._hparams.get("use_attn_dst", 1)))
        use_symmetric_norm = bool(int(self._hparams.get("use_symmetric_norm", 1)))
        edge_drop = float(self._hparams.get("edge_drop", 0.1))
        return RevGAT(
            in_dim,
            out_dim,
            out_dim,
            layers,
            heads,
            F.relu,
            dropout=self.dropout_ratio,
            attn_drop=0.0,
            edge_drop=edge_drop,
            use_attn_dst=use_attn_dst,
            use_symmetric_norm=use_symmetric_norm,
        )

    def _build_sgc(self, in_dim, out_dim, k):
        return nn.Sequential(
            SGConv(in_dim, out_dim, k=k, cached=True, bias=True),
            nn.ReLU(),
            nn.Dropout(self.dropout_ratio),
            nn.Linear(out_dim, out_dim),
        )

    def _get_fused_input_dim(self, hidden_dim):
        fusion = str(self.warmup_fusion).lower()
        if fusion == "concat":
            return hidden_dim + hidden_dim
        if fusion == "concat_absdiff":
            return hidden_dim * 3
        if fusion == "residual":
            return hidden_dim
        raise ValueError(f"Unknown warmup_fusion: {self.warmup_fusion}")

    def _fuse_after_warmup(self, text_base, image_base, enhanced_text, enhanced_image):
        fusion = str(self.warmup_fusion).lower()
        if fusion == "concat":
            return torch.cat([enhanced_text, enhanced_image], dim=-1)
        if fusion == "concat_absdiff":
            diff = torch.abs(enhanced_text - enhanced_image)
            return torch.cat([enhanced_text, enhanced_image, diff], dim=-1)
        if fusion == "residual":
            alpha = min(max(float(self.warmup_alpha), 0.0), 1.0)
            text = (1.0 - alpha) * text_base + alpha * enhanced_text
            image = (1.0 - alpha) * image_base + alpha * enhanced_image
            return text + image
        raise ValueError(f"Unknown warmup_fusion: {self.warmup_fusion}")

    def forward(self, graph, cross_graph, text_feat, image_feat):
        text_base = self.input_dropout(self.text_proj(text_feat))
        image_base = self.input_dropout(self.image_proj(image_feat))
        enhanced_text, enhanced_image, _ = self.cma(cross_graph, text_base, image_base)
        fused_feat = self._fuse_after_warmup(text_base, image_base, enhanced_text, enhanced_image)
        node_emb = self.gnn(graph, fused_feat)
        logits = self.classifier(self.dropout(node_emb))
        return logits
