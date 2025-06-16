import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
import torch

import torch.nn as nn
from layers.SEHS_layer import DataMerge
from layers.SEHS_layer import PatchEmbeddingWithPositionalEncoding
N=7
B=8
T=32

device = torch.device('cuda:4')
input=torch.randn(B,T,N).to(device)

dataMerge = DataMerge(device=device,seq_len=32,num_heads=4,num_layers=1)
patchEmbedding=PatchEmbeddingWithPositionalEncoding(patch_size=5,embed_dim=512,device=device)

output=patchEmbedding(input)

print(output.size())