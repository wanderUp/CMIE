import torch
import torch.nn as nn


# class SEHS_layer(nn.Module):
#     def __init__(self, device, num_layers, patch_size, num_heads, embed_dim, pred_len,seq_len):
#         super().__init__()
#         self.device = device
#         self.num_layers = num_layers
#         self.embed_dim = embed_dim
#         self.patch_size = patch_size
#         self.num_heads = num_heads
#         self.pred_len = pred_len
#
#         # 创建 PatchEmbeddingWithPositionalEncoding和SeriesEmbeddingWithPositionalEncoding 实例
#         self.patch_embed_pos = PatchEmbeddingWithPositionalEncoding(patch_size=self.patch_size,
#                                                                     embed_dim=self.embed_dim, device=self.device)
#         self.series_embed_pos = SeriesEmbeddingWithPositionalEncoding(embed_dim=self.embed_dim, device=self.device)
#
#         decoderLayer=nn.TransformerDecoderLayer(d_model=self.embed_dim, nhead=self.num_heads)
#         self.transformer_decoder=nn.TransformerDecoder(decoderLayer, num_layers=self.num_layers)
#         #self.transformer_decoder=decoderLayerWithoutSelfAttention(d_model=self.embed_dim, nhead=self.num_heads)
#
#         self.to(self.device)
#
#     def forward(self, x):
#         x = x.to(self.device)
#         B, T, N = x.shape;
#         outputs_list = []
#         for i in range(N):
#             Endogenous = x[:, :, i].unsqueeze(-1)
#             Exogenous = torch.cat((x[:, :, :i], x[:, :, i + 1:]), dim=2)
#
#             Endogenous = self.patch_embed_pos(Endogenous)  # (B, num_patches, embed_dim)
#             Exogenous = self.series_embed_pos(Exogenous)  # (B, T, embed_dim)
#
#             Endogenous = Endogenous.permute(1, 0, 2)
#             Exogenous = Exogenous.permute(1, 0, 2)
#
#             output = self.transformer_decoder(Endogenous, Exogenous)  # (num_patch,B,embed_dim)
#
#             output = output.permute(1, 0, 2)  # (B,num_patch,embed_dim)
#
#             outputs_list.append(output)
#         return torch.cat(outputs_list, dim=1)  # (B,num_patch*N,embed_dim)


class SEHS_layer(nn.Module):
    def __init__(self, device, num_layers, num_heads, embed_dim, seq_len):
        super().__init__()
        self.device = device

        self.dataMerge = [DataMerge(device=device, seq_len=seq_len, num_heads=num_heads, num_layers=num_layers)
                          for i in range(1)]
        self.patch_embed_pos = PatchEmbeddingWithPositionalEncoding(embed_dim=embed_dim, device=self.device)
        self.to(self.device)

    def forward(self, x):
        x = x.to(self.device)
        for item in self.dataMerge:
            x = item(x)
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
        self.transformer_decoder = nn.TransformerDecoder(decoderLayer, num_layers=self.num_layers)
        for param in self.transformer_decoder.parameters():
            param.requires_grad = False

        self.to(device)

    def forward(self, x):
        x = x.to(self.device)
        B, T, N = x.shape
        outputs_list = []
        for i in range(N):
            Endogenous = x[:, :, i].unsqueeze(-1)
            Exogenous = torch.cat((x[:, :, :i], x[:, :, i + 1:]), dim=2)

            Endogenous = Endogenous.permute(2, 0, 1)  # （1,B,T)
            Exogenous = Exogenous.permute(2, 0, 1)  # (N-1,B,T)

            output = self.transformer_decoder(Endogenous, Exogenous)  # (1,B,T)

            output = output.permute(1, 2, 0)  # (B,T,1)

            outputs_list.append(output)
        return torch.cat(outputs_list, dim=2) + x  # (B,T,N)


class PatchEmbeddingWithPositionalEncoding(nn.Module):
    def __init__(self, embed_dim, device):
        super(PatchEmbeddingWithPositionalEncoding, self).__init__()
        self.embed_dim = embed_dim
        self.device = device

        self.dropout = nn.Dropout(p=0.3)
        self.proj = None
        self.to(device)

    def forward(self, x):
        B, T, C = x.shape
        if self.proj is None:
            self.proj = nn.Conv1d(C, self.embed_dim, kernel_size=T, stride=T, device=self.device)
        x = x.permute(0, 2, 1)
        x = self.proj(x)
        x = self.dropout(x)
        x = x.permute(0, 2, 1)
        return x  # (B, num_patches, embed_dim)


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
