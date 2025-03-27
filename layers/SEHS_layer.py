import torch
import torch.nn as nn


class SEHS_layer(nn.Module):
    def __init__(self, device, num_layers, patch_size, num_heads, embed_dim, pred_len,limit_len):
        super().__init__()
        self.device = device
        self.num_layers = num_layers
        self.embed_dim = embed_dim
        self.patch_size = patch_size
        self.num_heads = num_heads
        self.pred_len = pred_len
        self.limit_len=limit_len
        # 创建 PatchEmbeddingWithPositionalEncoding和SeriesEmbeddingWithPositionalEncoding 实例
        self.patch_embed_pos = PatchEmbeddingWithPositionalEncoding(patch_size=self.patch_size,                                                                  embed_dim=self.embed_dim, device=self.device)
        self.series_embed_pos = SeriesEmbeddingWithPositionalEncoding(embed_dim=self.embed_dim, device=self.device)

        decoder_layer = nn.TransformerDecoderLayer(d_model=self.embed_dim, nhead=self.num_heads)
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=self.num_layers)
        self.AlignLinear=nn.LazyLinear(self.limit_len)
        self.to(self.device)


    def forward(self, x):
        x = x.to(self.device)
        B, T, N = x.shape;
        outputs_list = []
        for i in range(N):
            Endogenous = x[:, :, i].unsqueeze(-1)
            Exogenous = torch.cat((x[:, :, :i], x[:, :, i + 1:]), dim=2)


            Endogenous = self.patch_embed_pos(Endogenous)  # (B, num_patches, embed_dim)
            Exogenous = self.series_embed_pos(Exogenous)  # (B, T, embed_dim)


            Endogenous = Endogenous.permute(1, 0, 2)
            Exogenous = Exogenous.permute(1, 0, 2)


            output = self.transformer_decoder(Endogenous, Exogenous)  # (num_patch,B,embed_dim)


            output = output.permute(1, 2, 0)  # (B,embed_dim,num_patch)

            outputs_list.append(output)

        output=torch.cat(outputs_list, dim=2)# (B,embed_dim,num_patch*N)
        output=self.AlignLinear(output)
        output=output.permute(0, 2, 1)
        return   output

class PatchEmbeddingWithPositionalEncoding(nn.Module):
    def __init__(self, patch_size, embed_dim, device):
        super(PatchEmbeddingWithPositionalEncoding, self).__init__()
        self.pos_embed = None
        self.proj = None
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.device = device
        self.dropout = nn.Dropout(p=0.2)
        self.to(device)

    def forward(self, x):
        B, T, C = x.shape
        num_patches = T // self.patch_size
        if self.proj is None:
            self.proj = nn.Linear(self.patch_size * C, self.embed_dim).to(self.device)  # 线性映射
        if self.pos_embed is None:
            self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, self.embed_dim)).to(self.device)  # 可学习的位置编码
        # 切分时间步为多个 patches
        x = x.unfold(1, self.patch_size, self.patch_size)
        x = x.contiguous().view(B, num_patches, self.patch_size * C)

        # 通过线性层将每个 patch 映射到嵌入空间
        x = self.proj(x)
        x = self.dropout(x)
        # 添加位置编码
        x = x + self.pos_embed
        return x  # (B, seq_len//patch_size, embed_dim)


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
