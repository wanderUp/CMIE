from .losses import mape_loss, mase_loss, smape_loss
from .similar_utils import *


class decay_Loss(nn.Module):
    def __init__(self):
        """
        自定义损失函数：
        - 对每个样本计算 MAE，并对每一项乘以 1/sqrt(i)，其中 i 是时间步的索引。
        """
        super(decay_Loss, self).__init__()

    def forward(self, y_true, y_pred):
        """
        前向传播计算损失。
        输入形状为 (batch_size, seq_len, num_channels)。
        """
        batch_size, seq_len, num_channels = y_true.shape

        # 计算绝对误差 (batch_size, seq_len, num_channels)
        absolute_error = torch.abs(y_true - y_pred)

        # 计算权重项 1/sqrt(i) (seq_len,)
        weights = 1.0 / torch.sqrt(torch.arange(1, seq_len + 1, dtype=torch.float32, device=y_true.device))

        # 扩展权重以匹配绝对误差的形状 (1, seq_len, 1)
        weights = weights.view(1, seq_len, 1)

        # 加权后的误差 (batch_size, seq_len, num_channels)
        weighted_error = weights * absolute_error

        # 计算每个样本的损失 (batch_size,)
        loss_per_sample = torch.mean(weighted_error, dim=(1, 2))

        # 计算整个批次的平均损失
        loss = torch.mean(loss_per_sample)
        return loss


loss_dict = {
    "l1": nn.L1Loss(),
    "smooth_l1": nn.SmoothL1Loss(),
    "ce": nn.CrossEntropyLoss(),
    "mse": nn.MSELoss(),
    "smape": smape_loss(),
    "mape": mape_loss(),
    "mase": mase_loss(),
    "decay": decay_Loss()
}


class SEHSLoss(nn.Module):
    def __init__(self, task_name, output_loss):
        super(SEHSLoss, self).__init__()

        self.output_loss = loss_dict[output_loss]
        self.task_name = task_name

    def forward(self, outputs, batch_y, in_sample=None, freq_map=None, batch_y_mark=None):
        batch_y = batch_y.to(outputs.device)

        loss = self.output_loss(outputs, batch_y)

        return loss
