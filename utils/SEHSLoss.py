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


        batch_y = batch_y.to(outputs.device)

        loss = self.output_loss(outputs, batch_y)

        return loss
