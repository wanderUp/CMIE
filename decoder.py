import torch
import torch.nn as nn
from  layers.SEHS_layer import   PatchEmbeddingWithPositionalEncoding,SeriesEmbeddingWithPositionalEncoding
# 参数定义
batch_size = 32
seq_len_tgt = 1   # 目标序列长度
seq_len_src = 15  # 编码器的输出序列长度
d_model = 3       # 嵌入维度（模型维度）
num_layers = 2    # 解码器层数
num_heads = 1     # 多头注意力数量（必须能被 d_model 整除）


B=1
T=192
N=7


class SEHS_layer(nn.Module):
    def __init__(self, device, num_layers, patch_size, num_heads, embed_dim, pred_len):
        super().__init__()
        self.device = device
        self.num_layers = num_layers
        self.embed_dim = embed_dim
        self.patch_size = patch_size
        self.num_heads = num_heads
        self.pred_len = pred_len

        # 创建 PatchEmbeddingWithPositionalEncoding和SeriesEmbeddingWithPositionalEncoding 实例
        self.patch_embed_pos = PatchEmbeddingWithPositionalEncoding(patch_size=self.patch_size,
                                                                    embed_dim=self.embed_dim, device=self.device)
        self.series_embed_pos = SeriesEmbeddingWithPositionalEncoding(embed_dim=self.embed_dim, device=self.device)

        decoder_layer = nn.TransformerDecoderLayer(d_model=self.embed_dim, nhead=self.num_heads)
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=self.num_layers)
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

            output = output.permute(1, 0, 2)  # (B,num_patch,embed_dim)

            # output=output.reshape(output.shape[0],output.shape[1]*output.shape[2])#(B,num_patch*embed_dim)
            # self.recover=nn.Linear(output.shape[1],self.pred_len).to(self.device)
            # output=self.recover(output).unsqueeze(-1)#(B,T,1)

            outputs_list.append(output)
        return torch.cat(outputs_list, dim=1)  # (B,num_patch*N,embed_dim)


input = torch.randn(B, T, N)  # 生成随机输入
layers=SEHS_layer("cuda:0",1,8,2,768,192)
output = layers(input)

print(output.shape)  # 输出: [B, N, dim]