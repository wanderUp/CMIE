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


class CMIELoss(nn.Module):
    def __init__(self, task_name, output_loss, lambda_sim=0.0):
        super(CMIELoss, self).__init__()

        self.output_loss = loss_dict[output_loss]
        self.task_name = task_name
        self.lambda_sim = lambda_sim

    def _compute_l_sim(self, conv_embeds):
        """
        Compute embedding similarity regularization L_sim.
        conv_embeds: (B, N, D) where N=num_channels, D=embed_dim
        Paper Eq (8)-(10): L_sim = 2/(C*(C-1)) * sum_{i<j} e_i_tilde^T e_j_tilde
        """
        B, N, D = conv_embeds.shape
        # L2-normalize along embedding dimension
        e_norm = conv_embeds / (conv_embeds.norm(dim=2, keepdim=True) + 1e-8)
        # Pairwise cosine similarity: (B, N, N)
        sim_matrix = torch.bmm(e_norm, e_norm.transpose(1, 2))
        # Upper triangle (excluding diagonal), average over pairs and batch
        triu_mask = torch.triu(torch.ones(N, N, device=conv_embeds.device), diagonal=1).bool()
        pairwise_sim = sim_matrix[:, triu_mask]
        loss_sim = pairwise_sim.mean()
        return loss_sim

    def forward(self, outputs, batch_y, conv_embeds=None, in_sample=None, freq_map=None, batch_y_mark=None):
        batch_y = batch_y.to(outputs.device)

        loss = self.output_loss(outputs, batch_y)

        if self.lambda_sim > 0 and conv_embeds is not None:
            loss_sim = self._compute_l_sim(conv_embeds)
            loss = loss + self.lambda_sim * loss_sim

        return loss
