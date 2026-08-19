import torch
import torch.nn as nn
import numpy as np
from dgl.nn.functional import edge_softmax

class CrossModalGraphAttention(nn.Module):
    """
    基于图的稀疏跨模态注意力：
    - 内部增宽、外部维度不变：internal_dim = num_heads * head_dim
    - Q/K/V: hidden_dim -> internal_dim
    - OutProj: internal_dim -> hidden_dim
    - 注意力仅在图的边上计算，按源或目标节点做 edge softmax 归一化
    """
    def __init__(self, hidden_dim, num_heads, head_dim, dropout=0.5, attn_dropout=0.5, norm_by='dst'):
        super(CrossModalGraphAttention, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.internal_dim = num_heads * head_dim
        self.scale = np.sqrt(self.head_dim)
        self.norm_by = norm_by  # 'src' 或 'dst'

        # 文本查询图像（更新文本）
        self.text_to_image_q = nn.Linear(hidden_dim, self.internal_dim)
        self.text_to_image_k = nn.Linear(hidden_dim, self.internal_dim)
        self.text_to_image_v = nn.Linear(hidden_dim, self.internal_dim)
        self.text_out_proj = nn.Linear(self.internal_dim, hidden_dim)
        # 图像查询文本（更新图像）
        self.image_to_text_q = nn.Linear(hidden_dim, self.internal_dim)
        self.image_to_text_k = nn.Linear(hidden_dim, self.internal_dim)
        self.image_to_text_v = nn.Linear(hidden_dim, self.internal_dim)
        self.image_out_proj = nn.Linear(self.internal_dim, hidden_dim)

        # 门控残差
        self.text_gate = nn.Parameter(torch.tensor(0.1))
        self.image_gate = nn.Parameter(torch.tensor(0.1))

        self.dropout = nn.Dropout(dropout)
        self.attn_dropout = nn.Dropout(attn_dropout)
        self.layer_norm_text = nn.LayerNorm(hidden_dim)
        self.layer_norm_image = nn.LayerNorm(hidden_dim)

    def reset_parameters(self):
        for module in [self.text_to_image_q, self.text_to_image_k, self.text_to_image_v,
                       self.image_to_text_q, self.image_to_text_k, self.image_to_text_v,
                       self.text_out_proj, self.image_out_proj]:
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        nn.init.constant_(self.text_gate, 0.1)
        nn.init.constant_(self.image_gate, 0.1)

    def _reshape_heads(self, x):
        N = x.size(0)
        return x.view(N, self.num_heads, self.head_dim)

    def _graph_attention(self, graph, q_src, k_dst, v_dst):
        """
        图上稀疏注意力与聚合：
        - score_h = (q[u,h] · k[v,h]) / sqrt(d)
        - 对相同 src=u 的出边做 softmax（或 norm_by 指定）
        """
        device = q_src.device
        N = q_src.size(0)
        H = self.num_heads
        D = self.head_dim

        src, dst = graph.edges()  # [E], [E]
        scores = (q_src[src] * k_dst[dst]).sum(-1) / self.scale  # [E, H]
        attn = edge_softmax(graph, scores, norm_by=self.norm_by)  # [E, H]
        attn = self.attn_dropout(attn)
        msg = attn.unsqueeze(-1) * v_dst[dst]  # [E, H, D]
        out = torch.zeros((N, H * D), device=device, dtype=msg.dtype)
        out.index_add_(0, src, msg.reshape(-1, H * D))
        out = out.view(N, H, D)
        return out

    def forward(self, graph, text_emb, image_emb):
        # 文本查询图像（更新文本）
        text_q = self._reshape_heads(self.text_to_image_q(text_emb))
        image_k = self._reshape_heads(self.image_to_text_k(image_emb))
        image_v = self._reshape_heads(self.image_to_text_v(image_emb))
        text_from_image = self._graph_attention(graph, text_q, image_k, image_v)
        text_from_image = text_from_image.reshape(text_emb.size(0), self.internal_dim)
        text_from_image = self.text_out_proj(text_from_image)
        text_from_image = self.dropout(text_from_image)
        
        # 图像查询文本（更新图像）
        image_q = self._reshape_heads(self.image_to_text_q(image_emb))
        text_k = self._reshape_heads(self.text_to_image_k(text_emb))
        text_v = self._reshape_heads(self.text_to_image_v(text_emb))
        image_from_text = self._graph_attention(graph, image_q, text_k, text_v)
        image_from_text = image_from_text.reshape(image_emb.size(0), self.internal_dim)
        image_from_text = self.image_out_proj(image_from_text)
        image_from_text = self.dropout(image_from_text)
        
        # 门控残差 + LayerNorm
        enhanced_text = self.layer_norm_text(text_emb + self.text_gate * text_from_image)
        enhanced_image = self.layer_norm_image(image_emb + self.image_gate * image_from_text)
        return enhanced_text, enhanced_image, None
