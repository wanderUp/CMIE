import torch
import torch.nn as nn

from layers.SEHS_layer import SEHS_layer


class Model(nn.Module):
    def __init__(self, configs, device):
        super(Model, self).__init__()
        self.pred_len = configs.pred_len
        self.d_model = configs.d_model
        self.num_heads = configs.n_heads
        self.num_layers = configs.d_layers
        self.task_name = configs.task_name
        self.patch_size = configs.patch_size
        self.layers=SEHS_layer(device=device, embed_dim=self.d_model, num_layers=self.num_layers,
                                    num_heads=self.num_heads, patch_size=self.patch_size,pred_len=self.pred_len)

    def forecast(self, x):
        x=self.layers(x)
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
    def forward(self, x, mask=None):
        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            output = self.forecast(x)
        if self.task_name == 'classification':
            output = self.classification(x)
        if self.task_name == "imputation":
            output = self.imputation(x, mask)
        if self.task_name == "anomaly_detection":
            output = self.anomaly_detection(x)
        return output
