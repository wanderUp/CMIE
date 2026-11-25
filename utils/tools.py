import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import Normalize
from tqdm import tqdm
import seaborn as sns
from datetime import datetime
from distutils.util import strtobool
import pandas as pd
import torch.nn.functional as F
from utils.metrics import metric

plt.switch_backend('agg')

def adjust_learning_rate(optimizer, epoch, args):
    if args.lradj =='type1':
        lr_adjust = {epoch: args.learning_rate if epoch < 3 else args.learning_rate * (0.9 ** ((epoch - 3) // 1))}
    elif args.lradj =='type2':
        lr_adjust = {epoch: args.learning_rate * (args.decay_fac ** ((epoch - 1) // 1))}
    elif args.lradj =='type4':
        lr_adjust = {epoch: args.learning_rate * (args.decay_fac ** ((epoch) // 1))}
    else:
        args.learning_rate = 1e-4
        lr_adjust = {epoch: args.learning_rate if epoch < 3 else args.learning_rate * (0.9 ** ((epoch - 3) // 1))}
    print("lr_adjust = {}".format(lr_adjust))
    if epoch in lr_adjust.keys():
        lr = lr_adjust[epoch]
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        print('Updating learning rate to {}'.format(lr))


class EarlyStopping:
    def __init__(self, patience=7, verbose=False, delta=0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta

    def __call__(self, val_loss, model, path):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, path):
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), path + '/' + 'checkpoint.pth')
        self.val_loss_min = val_loss


class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


class StandardScaler():
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std+1e-6

    def transform(self, data):
        return (data - self.mean) / self.std

    def inverse_transform(self, data):
        return (data * self.std) + self.mean


def channel_stats_from_dataset(dataset):
    """从训练集提取逐通道全局统计量，shape (N, K)，原始尺度。

    K 列依次为: mean, std, min, max, skew, lag1_autocorr, slope.
    统计量在训练开始前一次性算好、固定不变（与 batch 无关），
    供 channel-aware prompt 使用。

    注：dataset.data_x 是标准化后的数据，这里先用 scaler 反变换回原始
    尺度，使所有统计量（包括新加的 min/max/skew/autocorr/slope）与
    mean/std 保持同一原始尺度，便于 LLM 理解真实数据分布。
    """
    # 原始尺度 mean / std 直接来自训练集拟合的 scaler
    mean = np.asarray(dataset.scaler.mean_)
    std = np.asarray(dataset.scaler.scale_)
    N = mean.shape[0]

    # 反变换回原始尺度，再算高阶统计量
    X_raw = np.asarray(dataset.data_x, dtype=np.float64) * std + mean  # (T, C)

    stats = np.zeros((N, 7))
    for i in range(N):
        x = X_raw[:, i]
        stats[i, 0] = mean[i]
        stats[i, 1] = std[i]
        stats[i, 2] = np.min(x)
        stats[i, 3] = np.max(x)
        stats[i, 4] = _skewness(x)
        stats[i, 5] = _lag1_autocorr(x)
        stats[i, 6] = _linear_slope(x)
    return stats


def _skewness(x):
    """偏度 (Fisher-Pearson 标准化矩)。"""
    x = np.asarray(x, dtype=np.float64)
    if x.shape[0] < 3:
        return 0.0
    m2 = np.mean((x - x.mean()) ** 2)
    m3 = np.mean((x - x.mean()) ** 3)
    if m2 <= 0.0:
        return 0.0
    return float(m3 / m2 ** 1.5)


def _lag1_autocorr(x):
    """lag-1 自相关系数。"""
    x = np.asarray(x, dtype=np.float64)
    if x.shape[0] < 3:
        return 0.0
    xm = x - x.mean()
    var = np.sum(xm * xm)
    if var <= 0.0:
        return 0.0
    return float(np.sum(xm[:-1] * xm[1:]) / var)


def _linear_slope(x):
    """线性趋势斜率（对时间索引归一化，避免量纲依赖）。"""
    x = np.asarray(x, dtype=np.float64)
    n = x.shape[0]
    if n < 2:
        return 0.0
    t = np.arange(n, dtype=np.float64)
    t = t - t.mean()
    denom = np.sum(t * t)
    if denom <= 0.0:
        return 0.0
    return float(np.sum(t * (x - x.mean())) / denom)


def channel_groups_from_dataset(dataset, threshold):
    """按训练集通道间 |Pearson| 相关做凝聚层次聚类，distance_threshold 自适应簇数 M；
    返回逐通道 patch 归属 assign (C,) int。threshold ∈ (0,1)，越大簇越少、patch 越大。"""
    from sklearn.cluster import AgglomerativeClustering
    X = np.asarray(dataset.data_x)  # (T, C)
    C = X.shape[1]
    threshold = float(threshold)
    if C < 2 or threshold <= 0 or threshold >= 1:
        return np.arange(C)  # 不分组：每个通道自成一个 patch
    corr = np.corrcoef(X.T)  # (C, C)
    sim = np.abs(np.nan_to_num(corr, nan=0.0))
    dist = 1.0 - sim
    np.fill_diagonal(dist, 0.0)
    clustering = AgglomerativeClustering(
        n_clusters=None,
        metric='precomputed',
        linkage='average',
        distance_threshold=threshold,
        compute_distances=True,
    )
    labels = clustering.fit_predict(dist)
    if int(labels.max()) + 1 < 2:
        return np.arange(C)  # 坍缩成 1 簇 → 回退不分组
    return labels


def visual(true, preds=None, name='./pic/test.pdf'):
    """
    Results visualization
    """
    plt.figure()
    plt.plot(true, label='GroundTruth', linewidth=2)
    if preds is not None:
        plt.plot(preds, label='Prediction', linewidth=2)
    plt.legend()
    plt.savefig(name, bbox_inches='tight')


def convert_tsf_to_dataframe(
    full_file_path_and_name,
    replace_missing_vals_with="NaN",
    value_column_name="series_value",
):
    col_names = []
    col_types = []
    all_data = {}
    line_count = 0
    frequency = None
    forecast_horizon = None
    contain_missing_values = None
    contain_equal_length = None
    found_data_tag = False
    found_data_section = False
    started_reading_data_section = False

    with open(full_file_path_and_name, "r", encoding="cp1252") as file:
        for line in file:
            # Strip white space from start/end of line
            line = line.strip()

            if line:
                if line.startswith("@"):  # Read meta-data
                    if not line.startswith("@data"):
                        line_content = line.split(" ")
                        if line.startswith("@attribute"):
                            if (
                                len(line_content) != 3
                            ):  # Attributes have both name and type
                                raise Exception("Invalid meta-data specification.")

                            col_names.append(line_content[1])
                            col_types.append(line_content[2])
                        else:
                            if (
                                len(line_content) != 2
                            ):  # Other meta-data have only values
                                raise Exception("Invalid meta-data specification.")

                            if line.startswith("@frequency"):
                                frequency = line_content[1]
                            elif line.startswith("@horizon"):
                                forecast_horizon = int(line_content[1])
                            elif line.startswith("@missing"):
                                contain_missing_values = bool(
                                    strtobool(line_content[1])
                                )
                            elif line.startswith("@equallength"):
                                contain_equal_length = bool(strtobool(line_content[1]))

                    else:
                        if len(col_names) == 0:
                            raise Exception(
                                "Missing attribute section. Attribute section must come before data."
                            )

                        found_data_tag = True
                elif not line.startswith("#"):
                    if len(col_names) == 0:
                        raise Exception(
                            "Missing attribute section. Attribute section must come before data."
                        )
                    elif not found_data_tag:
                        raise Exception("Missing @data tag.")
                    else:
                        if not started_reading_data_section:
                            started_reading_data_section = True
                            found_data_section = True
                            all_series = []

                            for col in col_names:
                                all_data[col] = []

                        full_info = line.split(":")

                        if len(full_info) != (len(col_names) + 1):
                            raise Exception("Missing attributes/values in series.")

                        series = full_info[len(full_info) - 1]
                        series = series.split(",")

                        if len(series) == 0:
                            raise Exception(
                                "A given series should contains a set of comma separated numeric values. At least one numeric value should be there in a series. Missing values should be indicated with ? symbol"
                            )

                        numeric_series = []

                        for val in series:
                            if val == "?":
                                numeric_series.append(replace_missing_vals_with)
                            else:
                                numeric_series.append(float(val))

                        if numeric_series.count(replace_missing_vals_with) == len(
                            numeric_series
                        ):
                            raise Exception(
                                "All series values are missing. A given series should contains a set of comma separated numeric values. At least one numeric value should be there in a series."
                            )

                        all_series.append(pd.Series(numeric_series).array)

                        for i in range(len(col_names)):
                            att_val = None
                            if col_types[i] == "numeric":
                                att_val = int(full_info[i])
                            elif col_types[i] == "string":
                                att_val = str(full_info[i])
                            elif col_types[i] == "date":
                                att_val = datetime.strptime(
                                    full_info[i], "%Y-%m-%d %H-%M-%S"
                                )
                            else:
                                raise Exception(
                                    "Invalid attribute type."
                                )  # Currently, the code supports only numeric, string and date types. Extend this as required.

                            if att_val is None:
                                raise Exception("Invalid attribute value.")
                            else:
                                all_data[col_names[i]].append(att_val)

                line_count = line_count + 1

        if line_count == 0:
            raise Exception("Empty file.")
        if len(col_names) == 0:
            raise Exception("Missing attribute section.")
        if not found_data_section:
            raise Exception("Missing series information under data section.")

        all_data[value_column_name] = all_series
        loaded_data = pd.DataFrame(all_data)

        return (
            loaded_data,
            frequency,
            forecast_horizon,
            contain_missing_values,
            contain_equal_length,
        )


def MASE(x, freq, pred, true):
    masep = np.mean(np.abs(x[:, freq:] - x[:, :-freq]))
    return np.mean(np.abs(pred - true) / (masep + 1e-8))

def adjustment(gt, pred):
    anomaly_state = False
    for i in range(len(gt)):
        if gt[i] == 1 and pred[i] == 1 and not anomaly_state:
            anomaly_state = True
            for j in range(i, 0, -1):
                if gt[j] == 0:
                    break
                else:
                    if pred[j] == 0:
                        pred[j] = 1
            for j in range(i, len(gt)):
                if gt[j] == 0:
                    break
                else:
                    if pred[j] == 0:
                        pred[j] = 1
        elif gt[i] == 0:
            anomaly_state = False
        if anomaly_state:
            pred[i] = 1
    return gt, pred


def cal_accuracy(y_pred, y_true):
    return np.mean(y_pred == y_true)

class fig:
    def __init__(self):
        self.fig, self.axs = plt.subplots(2, 2, figsize=(18, 6))
        self.axs=self.axs.flatten()
        self.datas=[]

    def put(self,data):
        self.datas.append(data)

    def show(self):
        for index in range(len(self.datas)):
            print("index=",index)
            sns.heatmap(self.datas[index].T, ax=self.axs[index], cmap='viridis')
            self.axs[index].set_title(index)
            self.axs[index].set_xlabel('Time Step')
            self.axs[index].set_ylabel('Channel')

        plt.tight_layout()
        plt.savefig("pic.png")

    def color(self):
        data = ((self.datas[0] - self.datas[1]) ** 2).mean(dim=0)
        data=np.array(data.cpu())
        print(data)
        norm = Normalize(vmin=np.min(data), vmax=np.max(data))
        cmap = cm.coolwarm
        colors = cmap(norm(data))  # shape (len(data), 4)

        # 步骤2：reshape 为 2D (1行N列)，供 imshow 使用
        colors_2d = colors.reshape(1, -1, 4)

        # 步骤3：画图（细色条 + 插值 + 坐标轴）
        fig, ax = plt.subplots(figsize=(8, 1.2))  # 图像瘦长一些
        im = ax.imshow(colors_2d, aspect='auto', interpolation='bicubic')

        # 步骤4：设置横坐标
        ax.set_xticks(np.arange(len(data)))  # 每个数据点一个刻度
        ax.set_xticklabels([str(i) for i in range(len(data))], fontsize=8, rotation=0)

        # 步骤5：隐藏纵坐标，只保留横坐标
        ax.set_yticks([])
        ax.tick_params(axis='x', bottom=True, labelbottom=True)

        # 步骤6：去除边框
        for spine in ax.spines.values():
            spine.set_visible(False)

        plt.tight_layout()
        plt.savefig("color.png")
        exit()
