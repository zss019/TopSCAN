from dataclasses import dataclass, field
from typing import Dict, Optional
import torch
import dgl


@dataclass
class EnhancedGraphPack:
    graph_for_backbone: dgl.DGLGraph
    edge_weight: Optional[torch.Tensor]
    edge_meta: Dict[str, torch.Tensor]
    reg_loss: torch.Tensor
    stats: Dict[str, float] = field(default_factory=dict)


@dataclass
class BackboneOutput:
    logits: torch.Tensor
    text_hidden: torch.Tensor
    image_hidden: torch.Tensor
    aux: Dict[str, torch.Tensor] = field(default_factory=dict)


@dataclass
class LossPack:
    total_loss: torch.Tensor
    cls_loss: torch.Tensor
    cm_loss: torch.Tensor
    struct_loss: torch.Tensor
    items: Dict[str, float] = field(default_factory=dict)
