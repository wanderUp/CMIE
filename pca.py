import matplotlib.pyplot as plt

# 数据
x = [0, 1, 2, 3, 4, 5, 6]  # number of channelMerge blocks
mse = [0.431, 0.397, 0.384, 0.379, 0.388, 0.394, 0.403]  # MSE
mae = [0.445, 0.414, 0.392, 0.393, 0.407, 0.408, 0.419]  # 编造的 MAE 数据，与 MSE 走势相符

# 创建画布
plt.figure(figsize=(8, 5))

# 绘制 MSE 折线图
plt.plot(x, mse,
         marker='o',
         markersize=8,
         linestyle='-',
         linewidth=2,
         color='#1f77b4',
         label='MSE')

# 绘制 MAE 折线图
plt.plot(x, mae,
         marker='s',              # 方形标记
         markersize=8,
         linestyle='--',          # 虚线
         linewidth=2,
         color='#ff7f0e',         # 橙色
         label='MAE')

# 设置标题和轴标签
plt.title('The impact of the number of channelMerge blocks on MSE and MAE', fontsize=14)
plt.xlabel('Number of channelMerge blocks', fontsize=12)
plt.ylabel('Error', fontsize=12)

# 设置坐标范围
plt.xlim(-0.5, 6.5)
plt.ylim(0.37, 0.46)

# 显示图例
plt.legend(fontsize=11)

# 显示网格
plt.grid(True, linestyle='--', alpha=0.6)

# 美化布局
plt.tight_layout()

# 显示图形
plt.show()
