import torch
import torch.nn as nn
from torch import relu
from utils.tools import fig


class CMIE_layer(nn.Module):
    def __init__(self, device, num_layers, num_heads, embed_dim, seq_len,merge_layers, use_ffc=True):
        super().__init__()
        self.device = device

        self.dataMerge = nn.ModuleList([DataMerge(device=device, seq_len=seq_len, num_heads=num_heads, num_layers=num_layers)
                          for i in range(merge_layers)])
        if use_ffc:
            self.patch_embed_pos = convolutionalEmbedding(embed_dim=embed_dim, seq_len=seq_len, device=self.device)
        else:
            self.patch_embed_pos = LinearEmbedding(embed_dim=embed_dim, seq_len=seq_len, device=self.device)
        self.to(self.device)

    def forward(self, x, patch_assign=None):
        x = x.to(self.device)
        for item in self.dataMerge:
            x = item(x, patch_assign)
        B, T, N = x.shape
        output = []
        for i in range(N):
            output.append(self.patch_embed_pos(x[:, :, i].unsqueeze(-1)))
        return torch.cat(output, dim=1)  # (B,N,dim)


class DataMerge(nn.Module):
    def __init__(self, device, seq_len, num_heads, num_layers):
        super(DataMerge, self).__init__()
        self.device = device
        self.seq_len = seq_len
        self.num_heads = num_heads
        self.num_layers = num_layers
        decoderLayer = nn.TransformerDecoderLayer(d_model=self.seq_len, nhead=self.num_heads)

        self.linear=nn.LazyLinear(2*seq_len)
        self.linear2=nn.LazyLinear(seq_len)

        self.transformer_decoder = nn.TransformerDecoder(decoderLayer, num_layers=self.num_layers)
        # 解冻两个注意力（self_attn 组内 + multihead_attn 组间 cross-attention）以实现自适应通道融合；
        # 内部 FFN（linear1/linear2）保持冻结——外部共享 FFN（self.linear/self.linear2）承担非线性映射。
        for name, param in self.transformer_decoder.named_parameters():
            if 'linear1' in name or 'linear2' in name:
                param.requires_grad = False
            else:
                param.requires_grad = True

        self.to(device)

    def forward(self, x, patch_assign=None):
        x = x.to(self.device)
        B, T, N = x.shape
        if patch_assign is not None and bool((patch_assign >= 0).all()):
            patch_tokens = self._build_patch_tokens(x, patch_assign)  # (M, B, T)
            M = patch_tokens.shape[0]
            use_patch = True
        else:
            use_patch = False
        if use_patch:
            # 以组为单位：循环 M 次（而非 N 次），组内通道保留各自维度拼成 query，
            # 与其他 M-1 组的池化 token 做 cross-attention。分组在进入 ChannelMerge
            # 前固定（patch_assign 是训练前算好的），这里只复用。每组输出按原始
            # 通道索引回写，保证残差连接前后的通道位置一致。
            output = torch.empty_like(x)
            groups = [torch.where(patch_assign == g)[0] for g in range(M)]
            for g in range(M):
                idx = groups[g]                        # (p_g,)
                Endogenous = x[:, :, idx].permute(2, 0, 1)  # (p_g, B, T) 组内所有通道作 query
                Exogenous = patch_tokens[torch.arange(M, device=self.device) != g]  # (M-1, B, T)
                group_output = self.transformer_decoder(Endogenous, Exogenous)  # (p_g, B, T)
                group_output = group_output.permute(1, 2, 0)       # (B, T, p_g)
                output[:, :, idx] = group_output
        else:
            # no-patch：以通道为单位循环 N 次，两两 attention (O(C^2))
            outputs_list = []
            for i in range(N):
                Endogenous = x[:, :, i].unsqueeze(-1)      # (B, T, 1)
                Exogenous = torch.cat((x[:, :, :i], x[:, :, i + 1:]), dim=2)  # (B, T, N-1)
                Exogenous = Exogenous.permute(2, 0, 1)     # -> (N-1, B, T)
                Endogenous = Endogenous.permute(2, 0, 1)   # （1,B,T)
                output = self.transformer_decoder(Endogenous, Exogenous)  # (1,B,T)
                output = output.permute(1, 2, 0)           # (B,T,1)
                outputs_list.append(output)
            output = torch.cat(outputs_list, dim=2)
        output = output + x  # (B,T,N)
        output = output.permute(0, 2, 1)
        mid = self.linear(output)
        mid = relu(mid)
        output = self.linear2(mid) + output
        output = output.permute(0, 2, 1)
        return output

    def _build_patch_tokens(self, x, patch_assign):
        # x: (B, T, N); patch_assign: (N,) long; 每个 patch 的成员通道在通道维 mean-pool 成 1 个 token -> (M, B, T)
        B, T, N = x.shape
        M = int(patch_assign.max().item()) + 1
        tokens = []
        for j in range(M):
            mask = (patch_assign == j)
            tokens.append(x[:, :, mask].mean(dim=2))  # (B, T)
        return torch.stack(tokens, dim=0)  # (M, B, T)


class convolutionalEmbedding(nn.Module):
    def __init__(self, embed_dim, seq_len, device):
        super(convolutionalEmbedding, self).__init__()
        self.embed_dim = embed_dim
        self.device = device
        self.seq_len = seq_len

        self.dropout = nn.Dropout(p=0.3)
        self.proj = nn.Conv1d(1, self.embed_dim, kernel_size=seq_len, stride=seq_len)
        self.pos_embedding = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
        self.to(device)

    def forward(self, x):
        B, T, C = x.shape
        x = x.permute(0, 2, 1)
        x = self.proj(x)
        x = self.dropout(x)
        x = x.permute(0, 2, 1)
        x = x + self.pos_embedding
        return x  # (B, 1, embed_dim)


class LinearEmbedding(nn.Module):
    """Replace FFC (Conv1d) with simple Linear projection — for w/o FFC ablation."""
    def __init__(self, embed_dim, seq_len, device):
        super(LinearEmbedding, self).__init__()
        self.embed_dim = embed_dim
        self.device = device
        self.seq_len = seq_len
        self.proj = nn.Linear(seq_len, embed_dim)
        self.pos_embedding = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.dropout = nn.Dropout(p=0.3)
        self.to(device)

    def forward(self, x):
        # x: (B, T, C=1)
        B, T, C = x.shape
        x = x.view(B, -1)           # (B, T)
        x = self.proj(x)             # (B, embed_dim)
        x = self.dropout(x)
        x = x.unsqueeze(1)           # (B, 1, embed_dim)
        x = x + self.pos_embedding
        return x  # (B, 1, embed_dim)


class SeriesEmbeddingWithPositionalEncoding(nn.Module):
    def __init__(self, embed_dim, device):
        super(SeriesEmbeddingWithPositionalEncoding, self).__init__()
        self.pos_embed = None
        self.proj = None
        self.embed_dim = embed_dim
        self.device = device
        self.dropout = nn.Dropout(p=0.2)
        self.to(device)

    def forward(self, x):
        B, T, N = x.shape  # 输入的形状为 (B, T, N)
        if self.proj is None:
            self.proj = nn.Linear(N, self.embed_dim).to(self.device)  # 线性映射，将每个时间步的 N 个特征映射到 embed_dim 空间
        if self.pos_embed is None:
            self.pos_embed = nn.Parameter(torch.zeros(1, T, self.embed_dim)).to(self.device)  # 可学习的位置编码
        x = self.proj(x)  # 通过线性层进行嵌入，输出的形状为 (B, T, embed_dim)
        x = self.dropout(x)
        x = x + self.pos_embed  # 添加位置编码，结果的形状是 (B, T, embed_dim)
        return x  # (B, T, embed_dim)
