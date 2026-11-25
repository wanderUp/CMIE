import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib import cm

# 示例一维数据（替换为你的数据）
data = np.array([0.12, 0.32, 0.55, 0.20, 0.91, 0.73, 0.65, 0.25, 0.1, 0.4])

# 步骤1：归一化 + colormap 映射
norm = Normalize(vmin=np.min(data), vmax=np.max(data))
cmap = cm.coolwarm
colors = cmap(data)  # shape (len(data), 4)

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
ax.set_xlabel("Index", fontsize=10)

# 步骤6：去除边框
for spine in ax.spines.values():
    spine.set_visible(False)

plt.tight_layout()
plt.show()
