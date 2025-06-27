import torch
import torch.nn as nn
from peft import get_peft_model, LoraConfig, TaskType
from transformers import GPT2Tokenizer

from layers.SEHS_layer import SEHS_layer
from models.GPT2_arch import AccustumGPT2Model
from utils.tools import StandardScaler


class Model(nn.Module):
    def __init__(self, configs, device):
        super(Model, self).__init__()

        self.description = configs.content
        self.N = configs.N
        self.pred_len = configs.pred_len
        self.num_layers = configs.d_layers
        self.top_k = configs.top_k
        self.task_name = configs.task_name
        self.device = device
        self.layers = SEHS_layer(device=device, embed_dim=configs.d_model, num_layers=self.num_layers,
                                 num_heads=configs.n_heads, seq_len=configs.seq_len, merge_layers=configs.merge_layers)

        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            r=configs.r,
            lora_alpha=configs.lora_alpha,
            lora_dropout=configs.lora_dropout,
            target_modules=["c_attn"]
        )

        # loads a pretrained GPT-2 base model
        self.llm_model = AccustumGPT2Model.from_pretrained('gpt2', output_attentions=True,
                                                           output_hidden_states=True)
        self.tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.llm_model.h = self.llm_model.h[:configs.gpt_layers]
        self.llm_model.config.num_hidden_layers = self.num_layers = configs.gpt_layers

        self.llm_model = get_peft_model(self.llm_model, peft_config)

        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model_dim = self.llm_model.config.hidden_size

        self.AlignDim = nn.Linear(configs.d_model, self.model_dim)

        self.prediction = prediction(configs.pred_len, configs.N)
        self.dropout = nn.Dropout(configs.dropout)
        self.to(device)

    def forecast(self, x):
        B, T, N = x.shape
        # 求 x 的均值和标准差（按 dim=1 进行）
        mean = x.mean(dim=1, keepdim=True)
        std = x.std(dim=1, keepdim=True)
        x_norm = StandardScaler(mean, std)
        x = x_norm.transform(x)

        #x_enc = x.permute(0, 2, 1).contiguous().reshape(B * N, T, 1)
        #lags = self.calcute_lags(x_enc)

        prompts = []
        # 遍历 N 个变量
        for i in range(N):
            xi = x[:, :, i]  # (B, T)

            # 提取统计特征，例如 mean/std/min/max/last 等
            mean = xi.mean(dim=1)
            std = xi.std(dim=1)
            last_val = xi[:, -1]

            # 构造 batch 中每个样本的 prompt（自然语言）
            for b in range(B):
                prompt_text = (
                    "These are some independent variables"
                    f"Variable {i + 1}: mean={mean[b]:.2f}, std={std[b]:.2f}, last={last_val[b]:.2f}. "
                    f"Forecast the next {self.pred_len} steps."
                )
                prompts.append(prompt_text)

        prompts = [prompts[i * N:(i + 1) * N] for i in range(B)]
        batch_prompts = [" ".join(batch_prompt) for batch_prompt in prompts]

        # Tokenize & embed
        tokenized = self.tokenizer(batch_prompts, return_tensors="pt", padding=True, truncation=True,
                                   max_length=1024).to(self.device)
        prompt_ids = tokenized.input_ids  # (B, len)
        prompt_embeds = self.llm_model.get_input_embeddings()(prompt_ids)  # (B, len, dim)

        prompt_token = prompt_embeds.shape[1]

        x = self.layers(x)

        prompt_embeddings = torch.nn.functional.scaled_dot_product_attention(
            query=prompt_embeds,
            key=x,
            value=x
        )

        x = self.AlignDim(x)
        x = self.dropout(x)

        x = torch.cat((prompt_embeddings, x), dim=1)

        dec_out, _ = self.llm_model(inputs_embeds=x)
        prompt_out = dec_out[:, :prompt_token, :]  # (B,prompt_token,dim)
        dec_out = dec_out[:, prompt_token:, :]  # (B,num_patch*N,dim)

        dec_out = self.prediction(dec_out)
        dec_out = x_norm.inverse_transform(dec_out)
        return dec_out

    def classification(self, x):
        x = self.layers(x)
        return x

    def imputation(self, x, mask):
        x = self.layers(x)
        return x

    def anomaly_detection(self, x):
        x = self.layers(x)
        return x

    def forward(self, x, mask=None):
        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            output = self.forecast(x)
        if self.task_name == "few_shot_forecast" or self.task_name == "zero_shot_forecast":
            output = self.forecast(x)
        if self.task_name == 'classification':
            output = self.classification(x)
        if self.task_name == "imputation":
            output = self.imputation(x, mask)
        if self.task_name == "anomaly_detection":
            output = self.anomaly_detection(x)

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
        self.var = nn.LazyLinear(N)

    def forward(self, x):
        x = self.pre(x)
        x = x.permute(0, 2, 1)
        x = self.var(x)

        # x = x.permute(0, 2, 1)
        return x
