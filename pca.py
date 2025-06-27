import os

from transformers import GPT2Tokenizer

from models.GPT2_arch import AccustumGPT2Model

os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
import torch
import seaborn as sns
import matplotlib.pyplot as plt

import torch.nn as nn
from layers.SEHS_layer import DataMerge
from layers.SEHS_layer import PatchEmbeddingWithPositionalEncoding

N = 7
B = 8
T = 32


x = torch.randn((B, T, N))
original_data=x[0]

fig, axs = plt.subplots(1, 2, figsize=(18, 6))

# 原始数据
sns.heatmap(original_data.T, ax=axs[0], cmap='viridis')
axs[0].set_title('Original Data')
axs[0].set_xlabel('Time Step')
axs[0].set_ylabel('Channel')

fused_data=DataMerge(device="cpu",seq_len=T,num_heads=4,num_layers=1)(x)[0].detach().numpy()

# 融合后数据
sns.heatmap(fused_data.T, ax=axs[1], cmap='viridis')
axs[1].set_title('Fused Data')
axs[1].set_xlabel('Time Step')
axs[1].set_ylabel('Fused Channel')

plt.tight_layout()
plt.show()





print()
