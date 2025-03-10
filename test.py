import torch

from layers.SEHS_layer import SEHS_layer


batch_size = 32
d_model = 3       # 嵌入维度（模型维度）
num_layers = 2    # 解码器层数
num_heads = 1     # 多头注意力数量（必须能被 d_model 整除）
seq_len_src=32
device=torch.device('cuda:3' if torch.cuda.is_available() else 'cpu')
r= SEHS_layer(device=device,embed_dim=d_model,num_layers=num_layers,num_heads=num_heads,patch_size=8)
memory = torch.rand(batch_size,seq_len_src , d_model)  # 编码器输出 (32, 15, 3)
out=r(memory)
print(out.shape)
