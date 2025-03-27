import torch
import torch.nn as nn

from layers.SEHS_layer import SEHS_layer
from models.GPT2_arch import AccustumGPT2Model

class Model(nn.Module):
    def __init__(self, configs, device):
        super(Model, self).__init__()
        self.device = device
        self.pred_len = configs.pred_len
        self.seq_len=configs.seq_len
        self.d_model = configs.d_model
        self.num_heads = configs.n_heads
        self.num_layers = configs.d_layers
        self.task_name = configs.task_name
        self.patch_size = configs.patch_size
        self.layers=SEHS_layer(device=device, embed_dim=self.d_model, num_layers=self.num_layers,
                                    num_heads=self.num_heads, patch_size=self.patch_size,pred_len=self.pred_len)
        self.N=configs.N
        self.output_projection = FlattenHead(self.pred_len,
                                             head_dropout=configs.dropout)
        self.prediction = nn.Linear(self.pred_len, self.pred_len)
        self.dropout = nn.Dropout(configs.dropout)


        # self.llm_model=AccustumGPT2Model.from_pretrained("gpt2",output_attentions=True,output_hidden_states=True)
        # self.llm_model.h = self.llm_model.h[:configs.gpt_layers]
        # self.llm_model.config.num_hidden_layers = self.num_layers = configs.gpt_layers
        self.to(self.device)

    def forecast(self, x):

        # 求 x 的均值和标准差（按 dim=1 进行）
        endo_mean = x.mean(dim=1, keepdim=True)
        endo_std = x.std(dim=1, keepdim=True)
        x = (x - endo_mean) / (endo_std + 1e-8)

        x = self.layers(x)#(B,dim,num_patch,N)
        x=x.permute(0,3,1,2)
        x=self.output_projection(x)#(B,N,pre)
        x=self.dropout(x)
        #x,_=self.llm_model(inputs_embeds=x)
        x=self.prediction(x)
        x=x.permute(0,2,1)


        x = x * (endo_std + 1e-8) + endo_mean
        return x




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
            output = self.imputation(x)
        if self.task_name == "anomaly_detection":
            output = self.anomaly_detection(x)
        return output

class FlattenHead(nn.Module):
    def __init__(self, target_window, head_dropout=0):
        super().__init__()
        self.flatten = nn.Flatten(start_dim=-2)
        self.linear = nn.LazyLinear(target_window)
        self.dropout = nn.Dropout(head_dropout)

    def forward(self, x):

        x = self.flatten(x)
        x = self.linear(x)
        x = self.dropout(x)
        return x