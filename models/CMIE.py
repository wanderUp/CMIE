import numpy as np
import torch
import torch.nn as nn
from peft import get_peft_model, LoraConfig, TaskType
from transformers import GPT2Tokenizer

from layers.CMIE_layer import CMIE_layer
from models.GPT2_arch import AccustumGPT2Model
from utils.tools import StandardScaler


class SimpleTransformerEncoder(nn.Module):
    """A simple Transformer Encoder used as a drop-in replacement for GPT-2 backbone."""
    def __init__(self, d_model, nhead, num_layers, dropout=0.1):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead,
            dim_feedforward=4 * d_model,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, x):
        # Return (output, None) to match GPT-2's (output, hidden_states) interface
        return self.encoder(x), None


class Model(nn.Module):
    def __init__(self, configs, device):
        super(Model, self).__init__()

        self.N = configs.N
        self.register_buffer('channel_stats', torch.zeros(self.N, 7))  # 逐通道 [mean,std,min,max,skew,lag1_autocorr,slope]，训练集固定值
        self.register_buffer('patch_assign', torch.full((self.N,), -1, dtype=torch.long))  # 逐通道 patch 归属，-1=未启用
        self.pred_len = configs.pred_len
        self.num_layers = configs.d_layers
        self.top_k = configs.top_k
        self.task_name = configs.task_name
        self.device = device
        self.use_llm = getattr(configs, 'use_llm', True)
        self.use_channel_aware_prompt = getattr(configs, 'use_channel_aware_prompt', True)
        self.llm_backbone = getattr(configs, 'llm_backbone', 'gpt2')

        # CMIE_layer: ChannelMerge + Embedding (DataMerge controlled by merge_layers)
        self.layers = CMIE_layer(device=device, embed_dim=configs.d_model, num_layers=self.num_layers,
                                 num_heads=configs.n_heads, seq_len=configs.seq_len,
                                 merge_layers=configs.merge_layers, use_ffc=getattr(configs, 'use_ffc', True))

        if self.use_llm:
            if self.llm_backbone == 'gpt2':
                # Original GPT-2 setup with LoRA
                peft_config = LoraConfig(
                    task_type=TaskType.CAUSAL_LM,
                    inference_mode=False,
                    r=configs.r,
                    lora_alpha=configs.lora_alpha,
                    lora_dropout=configs.lora_dropout,
                    target_modules=["c_attn"]
                )
                self.llm_model = AccustumGPT2Model.from_pretrained('gpt2', output_attentions=True,
                                                                   output_hidden_states=True)
                self.tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
                self.tokenizer.pad_token = self.tokenizer.eos_token
                self.llm_model.h = self.llm_model.h[:configs.gpt_layers]
                self.llm_model.config.num_hidden_layers = configs.gpt_layers
                self.llm_model = get_peft_model(self.llm_model, peft_config)
                self.model_dim = self.llm_model.config.hidden_size  # 768
            else:  # transformer encoder
                self.tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
                self.tokenizer.pad_token = self.tokenizer.eos_token
                self.prompt_embedding = nn.Embedding(self.tokenizer.vocab_size, configs.d_model)
                self.transformer_encoder = SimpleTransformerEncoder(
                    d_model=configs.d_model, nhead=configs.n_heads,
                    num_layers=configs.gpt_layers, dropout=configs.dropout
                )
                self.model_dim = configs.d_model

            # Align dimensions if needed
            if configs.d_model != self.model_dim:
                self.AlignDim = nn.Linear(configs.d_model, self.model_dim)
            else:
                self.AlignDim = nn.Identity()
        else:
            # w/o LLM: bypass the LLM backbone entirely
            self.model_dim = configs.d_model
            self.AlignDim = nn.Identity()

        # --- Channel-aware prompt: fixed per-channel text embedding + linear fusion ---
        self._prompt_embeds_cache = None  # lazy cache (N, model_dim); invalidated when stats change
        self.prompt_proj = nn.Linear(2 * self.model_dim, self.model_dim)
        self.prompt_norm = nn.LayerNorm(self.model_dim)

        self.prediction = prediction(configs.pred_len, configs.N)
        self.dropout = nn.Dropout(configs.dropout)
        self.to(device)

    def set_channel_stats(self, stats):
        """Set fixed per-channel global statistics from the training set.
        stats: numpy array or tensor of shape (N, 7);
        columns = [mean, std, min, max, skew, lag1_autocorr, slope]."""
        if isinstance(stats, np.ndarray):
            stats = torch.from_numpy(stats)
        self.channel_stats.copy_(stats.float().to(self.channel_stats.device))
        self._prompt_embeds_cache = None  # statistics changed -> rebuild prompt text

    def set_channel_groups(self, assign):
        """Set fixed channel-to-patch assignment for ChannelMerge (from similarity clustering).
        assign: numpy array or tensor of shape (N,), values in [0, M)."""
        if isinstance(assign, np.ndarray):
            assign = torch.from_numpy(assign)
        self.patch_assign.copy_(assign.long().to(self.patch_assign.device))
        self._prompt_embeds_cache = None  # keep cache coherent across re-grouping

    def _build_channel_prompts(self):
        """One prompt string per channel, built from its fixed training-set statistics.

        Each string reports that channel's global time-domain statistics
        (mean, std, min, max, skewness, lag-1 autocorrelation, linear slope);
        values are computed once over the whole training set, so the texts are
        batch-independent and identical for every sample.
        """
        prompts = []
        for i in range(self.N):
            m, s, lo, hi, sk, ac, sl = (float(self.channel_stats[i, k]) for k in range(7))
            prompts.append(
                f"This is an independent channel. Its characteristics are: "
                f"mean={m:.2f}, std={s:.2f}, min={lo:.2f}, max={hi:.2f}, "
                f"skew={sk:.2f}, lag1_autocorr={ac:.2f}, slope={sl:.3f}. "
                f"Forecast the next {self.pred_len} steps."
            )
        return prompts

    def _build_generic_prompts(self):
        """Ablation variant: one generic prompt per channel, without channel-specific statistics."""
        text = f"This is an independent channel. Forecast the next {self.pred_len} steps."
        return [text] * self.N

    @torch.no_grad()
    def _get_channel_text_embeds(self):
        """Per-channel text embedding of shape (N, model_dim), tokenize -> embed -> masked mean pooling.

        Computed once from the fixed statistics and cached; the result is
        batch-independent and carries no gradient.
        """
        if self._prompt_embeds_cache is not None:
            return self._prompt_embeds_cache
        prompts = (self._build_channel_prompts() if self.use_channel_aware_prompt
                   else self._build_generic_prompts())
        tok = self.tokenizer(prompts, return_tensors="pt", padding=True,
                             truncation=True, max_length=64)
        ids = tok.input_ids.to(self.device)
        mask = tok.attention_mask.to(self.device).unsqueeze(-1).float()
        if self.llm_backbone == 'gpt2':
            emb = self.llm_model.get_input_embeddings()(ids)
        else:  # transformer — dedicated embedding layer
            emb = self.prompt_embedding(ids)
        emb = (emb * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)  # (N, model_dim)
        self._prompt_embeds_cache = emb.detach()
        return self._prompt_embeds_cache

    def _fuse_channel_prompt(self, x):
        """Linear fusion: each channel token is concatenated with its own prompt-text embedding and projected back."""
        t = self._get_channel_text_embeds().to(dtype=x.dtype, device=x.device).unsqueeze(0)  # (1,N,d)
        x = self.prompt_proj(torch.cat([x, t.expand_as(x)], dim=-1))
        return self.prompt_norm(x)

    def forecast(self, x):
        B, T, N = x.shape
        # Normalize
        mean = x.mean(dim=1, keepdim=True)
        std = x.std(dim=1, keepdim=True)
        x_norm = StandardScaler(mean, std)
        x = x_norm.transform(x)

        # ChannelMerge + Embedding (DataMerge controlled by merge_layers=0 for w/o CM)
        x = self.layers(x, self.patch_assign)  # (B, N, d_model)
        self.conv_embeds = x

        if self.use_llm:
            # Align channel representations, then fuse each channel with its own prompt text.
            x = self.AlignDim(x)
            x = self.dropout(x)
            x = self._fuse_channel_prompt(x)  # (B, N, model_dim)

            # Backbone forward: the input sequence length equals the number of channels
            if self.llm_backbone == 'gpt2':
                dec_out, _ = self.llm_model(inputs_embeds=x)
            else:  # transformer encoder
                dec_out, _ = self.transformer_encoder(x)
            # dec_out is already (B, N, model_dim)
        else:
            # w/o LLM: direct prediction from channel representations
            x = self.AlignDim(x)
            x = self.dropout(x)
            dec_out = x  # (B, N, d_model)

        dec_out = self.prediction(dec_out)
        dec_out = x_norm.inverse_transform(dec_out)
        return dec_out

    def forward(self, x, mask=None):
        if self.task_name in ('long_term_forecast', 'short_term_forecast',
                              'few_shot_forecast', 'zero_shot_forecast'):
            output = self.forecast(x)
        return output

    def calcute_lags(self, x_enc):
        q_fft = torch.fft.rfft(x_enc.permute(0, 2, 1).contiguous(), dim=-1)
        k_fft = torch.fft.rfft(x_enc.permute(0, 2, 1).contiguous(), dim=-1)
        res = q_fft * torch.conj(k_fft)
        corr = torch.fft.irfft(res, dim=-1)
        mean_value = torch.mean(corr, dim=1)
        _, lags = torch.topk(mean_value, self.top_k, dim=-1)
        return lags


class FlattenHead(nn.Module):
    def __init__(self, nf, target_window, head_dropout=0):
        super().__init__()
        self.flatten = nn.Flatten(start_dim=-2)
        self.linear = nn.Linear(nf, target_window)
        self.dropout = nn.Dropout(head_dropout)

    def forward(self, x):
        x = self.flatten(x)
        x = self.linear(x)
        x = self.dropout(x)
        return x


class prediction(nn.Module):
    def __init__(self, pre_len, N):
        super().__init__()
        self.pre = nn.LazyLinear(pre_len)
        self.relu = nn.ReLU()
        self.larger = nn.LazyLinear(2 * pre_len)
        self.var = nn.LazyLinear(N)

    def forward(self, x):
        x = self.larger(x)
        x = self.relu(x)
        x = self.pre(x)
        x = x.permute(0, 2, 1)
        x = self.var(x)
        return x
