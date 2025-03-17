import torch
import torch.nn as nn
from peft import get_peft_model, LoraConfig, TaskType
from transformers import GPT2Model, GPT2Tokenizer, GPT2Config

from layers.SEHS_layer import SEHS_layer
from models.GPT2_arch import AccustumGPT2Model
from utils.tools import StandardScaler


class Model(nn.Module):
    def __init__(self, configs, device):
        super(Model, self).__init__()
        self.device = device
        self.description = configs.content
        self.seq_len=configs.seq_len
        self.pred_len = configs.pred_len
        self.d_model = configs.d_model
        self.num_heads = configs.n_heads
        self.num_layers = configs.d_layers
        self.top_k = configs.top_k
        self.task_name = configs.task_name
        self.patch_size = configs.patch_size
        self.layers=SEHS_layer(device=device, embed_dim=self.d_model, num_layers=self.num_layers,
                                    num_heads=self.num_heads, patch_size=self.patch_size,pred_len=self.pred_len)


        self.gpt2_config = GPT2Config.from_pretrained('gpt2')
        self.gpt2_config.num_hidden_layers = configs.llm_layers
        self.gpt2_config.output_attentions = True
        self.gpt2_config.output_hidden_states = True
        self.dropout = nn.Dropout(configs.dropout)


        # peft_config = LoraConfig(
        #     task_type=TaskType.CAUSAL_LM,
        #     inference_mode=False,
        #     r=configs.r,
        #     lora_alpha=configs.lora_alpha,
        #     lora_dropout=configs.lora_dropout,
        #     target_modules=["c_attn"]
        # )
        try:
            self.llm_model = GPT2Model.from_pretrained(
                'gpt2',
                trust_remote_code=True,
                local_files_only=True,
                config=self.gpt2_config,
            )
        except EnvironmentError:  # downloads model from HF is not already done
            print("Local model files not found. Attempting to download...")
            self.llm_model = GPT2Model.from_pretrained(
                'gpt2',
                trust_remote_code=True,
                local_files_only=False,
                config=self.gpt2_config,
            )

        try:
            self.tokenizer = GPT2Tokenizer.from_pretrained(
                'gpt2',
                trust_remote_code=True,
                local_files_only=True
            )
        except EnvironmentError:  # downloads the tokenizer from HF if not already done
            print("Local tokenizer files not found. Atempting to download them..")
            self.tokenizer = GPT2Tokenizer.from_pretrained(
                'openai-community/gpt2',
                trust_remote_code=True,
                local_files_only=False
            )



        # self.gpt2 = AccustumGPT2Model.from_pretrained('gpt2', output_attentions=True,
        #                                               output_hidden_states=True)  # loads a pretrained GPT-2 base model
        # self.gpt2_text = AccustumGPT2Model.from_pretrained('gpt2', output_attentions=True,
        #                                                    output_hidden_states=True)  # loads a pretrained GPT-2 base model
        # self.tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        # self.tokenizer.pad_token = self.tokenizer.eos_token
        # self.gpt2.h = self.gpt2.h[:configs.gpt_layers]
        # self.gpt2_text.h = self.gpt2_text.h[:configs.gpt_layers]
        # self.gpt2 = get_peft_model(self.gpt2, peft_config)

        #word_embedding = torch.tensor(torch.load(configs.word_embedding_path)).to(device=device)

        self.to(self.device)

    def forecast(self, x):
        B,T,N=x.shape
        # 求 x 的均值和标准差（按 dim=1 进行）
        mean = x.mean(dim=1, keepdim=True)
        std = x.std(dim=1, keepdim=True)
        x_norm=StandardScaler(mean,std)
        x = x_norm.transform(x)


        x_enc = x.permute(0, 2, 1).contiguous().reshape(B * N, T, 1)



        lags = self.calcute_lags(x_enc)
        trends = x_enc.diff(dim=1).sum(dim=1)

        prompt = []
        for b in range(x_enc.shape[0]):
            lags_values_str = str(lags[b].tolist())
            prompt_ = (
                f"<|start_prompt|>Dataset description: {self.description}"
                f"Task description: forecast the next {str(self.pred_len)} steps given the previous {str(self.seq_len)} steps information; "
                "Input statistics: "
                f"the trend of input is {'upward' if trends[b] > 0 else 'downward'}, "
                f"top 5 lags are : {lags_values_str}<|<end_prompt>|>"
            )
            prompt.append(prompt_)

        x_enc = x_enc.reshape(B, N, T).permute(0, 2, 1).contiguous()
        self.tokenizer.pad_token = self.tokenizer.eos_token
        prompt = self.tokenizer(prompt, return_tensors="pt", padding=True, truncation=True, max_length=2048).input_ids
        prompt_embeddings = self.llm_model.get_input_embeddings()(prompt.to(x_enc.device))  # (batch, prompt_token, dim)
        prompt_embeddings=prompt_embeddings.reshape(B , -1, prompt_embeddings.shape[-1])
        self.Align=nn.Linear(self.d_model,prompt_embeddings.shape[-1]).to(x_enc.device)
        x = self.layers(x)
        x=self.Align(x);

        print(x.shape)
        print(prompt_embeddings.shape)
        x=torch.cat((prompt_embeddings,x),dim=1)

        print(x.shape)

        dec_out = self.llm_model(inputs_embeds=x).last_hidden_state
        print(dec_out.shape)
        dec_out = dec_out[:, :, :self.d_model]

        dec_out = torch.reshape((-1, N, dec_out.shape[-2], dec_out.shape[-1]))
        dec_out = dec_out.permute(0, 1, 3, 2).contiguous()

        output_projection = FlattenHead(self.d_model * self.patch_nums, self.pred_len,
                                             head_dropout=self.dropout)
        dec_out = output_projection(dec_out[:, :, :, -self.patch_nums:])

        dec_out = dec_out.permute(0, 2, 1).contiguous()
        print(dec_out.shape)


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
    def forward(self, x,mask=None):
        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
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