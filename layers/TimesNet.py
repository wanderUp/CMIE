import torch
import torch.nn as nn


class TimesBlock(nn.Module):
    def __init__(self, in_channels, periods):
        super(TimesBlock, self).__init__()
        self.periods = periods  # 不同周期窗口
        self.conv_layers = nn.ModuleList([
            nn.Conv2d(in_channels, in_channels, kernel_size=(1, 1)) for _ in range(len(periods))
        ])
        self.projection = nn.Conv1d(in_channels, in_channels, kernel_size=1)

    def forward(self, x):
        # x: (batch, seq_len, channels)
        B, L, C = x.shape
        x = x.permute(0, 2, 1)  # (batch, channels, seq_len)
        res = x

        period_outs = []
        for idx, period in enumerate(self.periods):
            # 补齐以适配周期
            pad_len = (period - L % period) % period
            if pad_len > 0:
                x_pad = torch.cat([x, x[:, :, -pad_len:]], dim=-1)  # (B, C, L + pad_len)
            else:
                x_pad = x

            # 重排，分块
            B_p, C_p, L_p = x_pad.shape
            new_len = L_p // period
            x_period = x_pad.reshape(B_p, C_p, new_len, period)

            # 2D 卷积（沿着周期卷积）
            out = self.conv_layers[idx](x_period)
            out = out.reshape(B, C, -1)[:, :, :L]  # 截断到原始长度
            period_outs.append(out)

        # 聚合不同周期
        out = sum(period_outs) / len(period_outs)
        out = out + res

        # 投影
        out = self.projection(out)
        out = out.permute(0, 2, 1)  # (B, L, C)
        return out


class TimesNet(nn.Module):
    def __init__(self, in_channels, model_dim, num_blocks=3):
        super(TimesNet, self).__init__()
        self.input_projection = nn.Linear(in_channels, model_dim)

        # 每层周期
        self.blocks = nn.ModuleList([
            TimesBlock(model_dim, periods=[2 ** (i + 1), 2 ** (i + 2)]) for i in range(num_blocks)
        ])
        self.norm = nn.LayerNorm(model_dim)

    def forward(self, x):
        """
        x: (batch, seq_len, in_channels)
        """
        # 投影到模型维度
        x = self.input_projection(x)

        # 逐层通过 TimesBlock
        for block in self.blocks:
            x = block(x)

        # 最后正则化
        x = self.norm(x)
        return x  # (batch, seq_len, model_dim)
