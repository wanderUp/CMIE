import matplotlib.pyplot as plt

# 数据
x = [0, 1, 2, 3, 4, 5, 6]  # number of dataMerge block
y = [0.431, 0.397, 0.384, 0.379, 0.388, 0.394, 0.403]  # MSE

# 创建画布和坐标轴
plt.figure(figsize=(8, 5))

# 绘制折线图
plt.plot(x, y,
         marker='o',         # 数据点标记为圆圈
         markersize=8,       # 标记大小
         linestyle='-',      # 实线连接
         linewidth=2,        # 线宽
         color='#1f77b4')    # 线条颜色（蓝色）

# 设置坐标轴标签和标题
plt.ylabel('MSE', fontsize=12)
plt.title('The impact of the number of dataMerge blocks on the MSE', fontsize=14)

# 设置横纵坐标范围
plt.xlim(-0.5, 6.5)  # 横轴多留一些空白
plt.ylim(0.37, 0.44)  # 纵轴范围包含所有数据点

# 显示网格线
plt.grid(True, linestyle='--', alpha=0.6)

# 显示图形
plt.tight_layout()  # 自动调整布局
plt.show()