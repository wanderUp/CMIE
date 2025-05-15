import torch
import torch.nn as nn
import torch.nn.functional as F

import torch.nn as nn
import torch.nn.functional as F

class decoderLayerWithoutSelfAttention(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1):
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, tgt, memory, tgt_mask=None, memory_mask=None,
                tgt_key_padding_mask=None, memory_key_padding_mask=None):
        # No self-attention

        # Cross-attention
        tgt2, _ = self.cross_attn(tgt, memory, memory,
                                  attn_mask=memory_mask,
                                  key_padding_mask=memory_key_padding_mask)
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)

        # Feed-forward
        tgt2 = self.linear2(self.dropout(F.relu(self.linear1(tgt))))
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)

        return tgt


class encoderLayerWithoutFFN(nn.Module):
    def __init__(self, embed_size, num_heads):
        super(encoderLayerWithoutFFN, self).__init__()
        # 使用 PyTorch 提供的 MultiheadAttention
        self.self_attn = nn.MultiheadAttention(embed_size, num_heads)
        self.norm = nn.LayerNorm(embed_size)

    def forward(self, src, src_mask=None):
        # 自注意力机制
        attn_output, _ = self.self_attn(src, src, src, attn_mask=src_mask)
        # 残差连接和层归一化
        out = self.norm(src + attn_output)
        return out