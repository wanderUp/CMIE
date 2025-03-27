import torch
import torch.nn.functional as F
import torch.nn as nn
from .similar_utils import *
from copy import deepcopy

from .losses import mape_loss, mase_loss, smape_loss

loss_dict = {
    "l1": nn.L1Loss(),
    "smooth_l1": nn.SmoothL1Loss(),
    "ce": nn.CrossEntropyLoss(),
    "mse": nn.MSELoss(),
    "smape": smape_loss(),
    "mape": mape_loss(),
    "mase": mase_loss(),
}


class SEHSLoss(nn.Module):
    def __init__(self, task_name,output_loss):
        super(SEHSLoss, self).__init__()

        self.output_loss = loss_dict[output_loss]
        self.task_name = task_name


    def forward(self, outputs, batch_y, in_sample=None, freq_map=None, batch_y_mark=None):

        output=outputs["output"]
        intermidiate_time=outputs["intermidiate_time"]
        intermidiate_prompt=outputs["intermidiate_prompt"]
        batch_y = batch_y.to(output.device)

        feature_loss = sum(
            [
                (0.8 ** idx) * self.output_loss(feat_time, feat_text)
                for idx, (feat_time, feat_text) in enumerate(
                zip(intermidiate_time[::-1], intermidiate_prompt[::-1])
            )
            ]
        )

        loss = self.output_loss(output, batch_y)+feature_loss

        return loss
