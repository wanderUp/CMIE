import torch
import torch.nn as nn
from peft import get_peft_model, LoraConfig, TaskType
from transformers import GPT2Model, GPT2Tokenizer, GPT2Config

from layers.SEHS_layer import SEHS_layer
from layers.TimesNet import TimesNet
from models.GPT2_arch import AccustumGPT2Model
from utils.tools import StandardScaler


class Model(nn.Module):
    def __init__(self, configs, device):
        super(Model, self).__init__()
        self.timesnet = None
        self.N=configs.N
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


        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            r=configs.r,
            lora_alpha=configs.lora_alpha,
            lora_dropout=configs.lora_dropout,
            target_modules=["c_attn"]
        )
        self.llm_model = AccustumGPT2Model.from_pretrained('gpt2', output_attentions=True,
                                                      output_hidden_states=True)  # loads a pretrained GPT-2 base model
        self.tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.llm_model.h = self.llm_model.h[:configs.gpt_layers]
        self.llm_model.config.num_hidden_layers = self.num_layers = configs.gpt_layers

        self.llm_model = get_peft_model(self.llm_model, peft_config)




        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model_dim = self.llm_model.config.hidden_size
        head_nf = (self.seq_len // self.patch_size) * self.N
        self.AlignDim = nn.Linear(self.d_model, self.model_dim)
        self.output_projection = FlattenHead(head_nf, self.pred_len,
                                             head_dropout=configs.dropout)
        self.prediction = nn.Linear(self.model_dim, self.N)
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
        if self.timesnet is None:
            self.timesnet=TimesNet(in_channels=N, model_dim=self.d_model).to(self.device)
        timesnet_output = self.timesnet(x)# (batch, seq_len, model_dim)
        cycle_features = timesnet_output.mean(dim=1)  # (batch, model_dim)
        cycle_features_list = cycle_features.tolist()
        prompt = []
        for b in range(x.shape[0]):
            lags_values_str = str(lags[b].tolist())
            cycle_str = ', '.join([f"{v:.4f}" for v in cycle_features_list[b]])
            prompt_ = (
                f"<|start_prompt|>Dataset description: {self.description}"
                f"Task: forecast next {str(self.pred_len)} steps; "
                "Input statistics: "
                f"Cycle features: {cycle_str}; "
                f"trend {'upward' if trends[b] > 0 else 'downward'}, "
                f"top 5 lags : {lags_values_str}<|<end_prompt>|>"
            )
            prompt.append(prompt_)


        prompt = self.tokenizer(prompt, return_tensors="pt", padding=True, truncation=True, max_length=512).input_ids
        prompt_embeddings = self.llm_model.get_input_embeddings()(prompt.to(self.device))  # (batch, prompt_token, dim)
        prompt_embeddings=prompt_embeddings.reshape(B , -1, self.model_dim)

        x = self.layers(x)
        x=self.AlignDim(x);

        # prompt_embeddings = torch.nn.functional.scaled_dot_product_attention(
        #     query=x,
        #     key=prompt_embeddings,
        #     value=prompt_embeddings
        # )

        prompt_token = prompt_embeddings.shape[1]
        x=torch.cat((prompt_embeddings,x),dim=1)

        dec_out,_ = self.llm_model(inputs_embeds=x)
        prompt_out = dec_out[:,:prompt_token,:]#(B,prompt_token,dim)
        dec_out = dec_out[:,prompt_token:,:]#(B,num_patch*N,dim)



        dec_out = dec_out.reshape((B, N, -1, self.model_dim))#(B,N,num_patch,dim)
        dec_out = dec_out.permute(0, 3, 1, 2).contiguous()#(B,dim,N,num_patch)
        dec_out = self.output_projection(dec_out)#(B,dim,pre_len)

        dec_out = dec_out.permute(0, 2, 1).contiguous()
        dec_out=self.prediction(dec_out)



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