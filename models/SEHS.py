import torch
import torch.nn as nn

from layers.SEHS_layer import SEHS_layer


class Model(nn.Module):
    def __init__(self, configs, device):
        super(Model, self).__init__()
        self.marklinear = None
        self.device = device
        self.pred_len = configs.pred_len
        self.d_model = configs.d_model
        self.num_heads = configs.n_heads
        self.num_layers = configs.d_layers
        self.task_name = configs.task_name
        self.patch_size = configs.patch_size
        self.layers=SEHS_layer(device=device, embed_dim=self.d_model, num_layers=self.num_layers,
                                    num_heads=self.num_heads, patch_size=self.patch_size,pred_len=self.pred_len)

        self.to(self.device)

    def forecast(self, x,mark):

        self.marklinear = nn.Linear(mark.shape[-1], x.shape[-1]).to(self.device)
        mark = self.marklinear(mark)
        x+=mark
        # 求 x 的均值和标准差（按 dim=1 进行）
        endo_mean = x.mean(dim=1, keepdim=True)
        endo_std = x.std(dim=1, keepdim=True)
        x = (x - endo_mean) / (endo_std + 1e-8)

        x = self.layers(x,)

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
    def forward(self, x, mark,mask=None):
        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            output = self.forecast(x,mark)
        if self.task_name == 'classification':
            output = self.classification(x,mark)
        if self.task_name == "imputation":
            output = self.imputation(x, mask)
        if self.task_name == "anomaly_detection":
            output = self.anomaly_detection(x,mark)
        return output
