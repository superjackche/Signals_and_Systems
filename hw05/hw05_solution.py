"""
信号与系统 - 作业5
卷积的数值计算
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams

# 设置中文显示
rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
rcParams['axes.unicode_minus'] = False

def plot_discrete_signal(n, x, title, xlabel='n', ylabel='幅值'):
    """绘制离散信号"""
    plt.figure(figsize=(10, 4))
    plt.stem(n, x, basefmt=' ')
    plt.grid(True, alpha=0.3)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.tight_layout()
    
def convolution(x, h):
    """计算卷积 y[n] = x[n] * h[n]"""
    return np.convolve(x, h)

# ==================== 任务 (1) ====================
print("=" * 60)
print("任务 (1): 生成信号 x[n] 和 h[n]，并画出相应的图形")
print("=" * 60)

# 定义信号 x[n] = n, 0 ≤ n ≤ 5
n_x = np.arange(0, 6)
x = n_x.copy()

# 定义单位脉冲响应 h[n] = 1, 0 ≤ n ≤ 5
n_h = np.arange(0, 6)
h = np.ones(6)

print(f"x[n] = {x}")
print(f"h[n] = {h}")

# 绘制 x[n]
plot_discrete_signal(n_x, x, 'x[n] = n, 0 ≤ n ≤ 5')
plt.savefig('d:/Code/Signals_and_Systems/hw05/x_n.png', dpi=300, bbox_inches='tight')
plt.show()

# 绘制 h[n]
plot_discrete_signal(n_h, h, 'h[n] = 1, 0 ≤ n ≤ 5')
plt.savefig('d:/Code/Signals_and_Systems/hw05/h_n.png', dpi=300, bbox_inches='tight')
plt.show()

# ==================== 任务 (2) ====================
print("\n" + "=" * 60)
print("任务 (2): 计算卷积 y[n] = x[n] * h[n]，并画出其图形")
print("=" * 60)

# 计算卷积
y = convolution(x, h)
n_y = np.arange(0, len(y))

print(f"y[n] = x[n] * h[n] = {y}")
print(f"卷积结果长度: {len(y)}")

# 绘制卷积结果
plot_discrete_signal(n_y, y, 'y[n] = x[n] * h[n] (卷积结果)')
plt.savefig('d:/Code/Signals_and_Systems/hw05/y_n_convolution.png', dpi=300, bbox_inches='tight')
plt.show()

# ==================== 任务 (3) ====================
print("\n" + "=" * 60)
print("任务 (3): 选择一段有意义的数据进行处理")
print("=" * 60)

# 这里我选择学生成绩数据作为示例
# 假设这是一个班级30名学生的考试成绩
np.random.seed(42)  # 设置随机种子以保证可重复性
scores = np.random.normal(75, 12, 30)  # 均值75，标准差12
scores = np.clip(scores, 0, 100)  # 限制在0-100之间
scores = np.round(scores, 1)  # 保留一位小数

print("学生成绩数据（共30名学生）:")
print(scores)

# (a) 显示该段数据，加注必要的修饰
plt.figure(figsize=(12, 5))
plt.plot(range(1, len(scores)+1), scores, 'o-', linewidth=2, markersize=6, label='学生成绩')
plt.xlabel('学生编号', fontsize=12)
plt.ylabel('成绩（分）', fontsize=12)
plt.title('班级学生成绩分布图', fontsize=14, fontweight='bold')
plt.grid(True, alpha=0.3, linestyle='--')
plt.legend(fontsize=10)
plt.axhline(y=60, color='r', linestyle='--', alpha=0.5, label='及格线')
plt.ylim(0, 105)
plt.tight_layout()
plt.savefig('d:/Code/Signals_and_Systems/hw05/scores_plot.png', dpi=300, bbox_inches='tight')
plt.show()

# (b) 显示统计分布图 (histogram)
plt.figure(figsize=(10, 6))
n_bins, bins, patches = plt.hist(scores, bins=10, edgecolor='black', alpha=0.7, color='skyblue')
plt.xlabel('成绩区间（分）', fontsize=12)
plt.ylabel('学生人数', fontsize=12)
plt.title('学生成绩分布直方图', fontsize=14, fontweight='bold')
plt.grid(True, alpha=0.3, axis='y')
plt.axvline(x=60, color='r', linestyle='--', linewidth=2, label='及格线')
plt.legend(fontsize=10)
plt.tight_layout()
plt.savefig('d:/Code/Signals_and_Systems/hw05/scores_histogram.png', dpi=300, bbox_inches='tight')
plt.show()

# (c) 计算平均值，最小值，最大值，中位数
mean_score = np.mean(scores)
min_score = np.min(scores)
max_score = np.max(scores)
median_score = np.median(scores)

print("\n统计数据:")
print(f"平均值: {mean_score:.2f}")
print(f"最小值: {min_score:.2f}")
print(f"最大值: {max_score:.2f}")
print(f"中位数: {median_score:.2f}")
print(f"标准差: {np.std(scores):.2f}")
print(f"及格人数: {np.sum(scores >= 60)}")
print(f"及格率: {np.sum(scores >= 60) / len(scores) * 100:.1f}%")

# (d) 利用 h[n] = 1/N[u[n] - u[n-N]] 处理此段数据
print("\n" + "=" * 60)
print("任务 (3d): 利用滑动平均滤波器处理数据")
print("=" * 60)

# 选择窗口大小 N = 5（5点滑动平均）
N = 5

# 创建滑动平均滤波器
h_filter = np.ones(N) / N
print(f"滤波器 h[n] = 1/{N} * [u[n] - u[n-{N}]]")
print(f"滤波器系数: {h_filter}")

# 对数据进行滤波（使用卷积）
scores_filtered = np.convolve(scores, h_filter, mode='valid')
n_filtered = np.arange(N-1, N-1+len(scores_filtered))

print(f"\n原始数据长度: {len(scores)}")
print(f"滤波后数据长度: {len(scores_filtered)}")

# 绘制处理前后的对比图
plt.figure(figsize=(14, 6))

# 原始数据
plt.subplot(1, 2, 1)
plt.plot(range(1, len(scores)+1), scores, 'o-', linewidth=2, markersize=6, alpha=0.7, label='原始成绩')
plt.xlabel('学生编号', fontsize=12)
plt.ylabel('成绩（分）', fontsize=12)
plt.title('原始学生成绩', fontsize=13, fontweight='bold')
plt.grid(True, alpha=0.3, linestyle='--')
plt.axhline(y=60, color='r', linestyle='--', alpha=0.5)
plt.legend(fontsize=10)
plt.ylim(0, 105)

# 滤波后数据
plt.subplot(1, 2, 2)
plt.plot(n_filtered+1, scores_filtered, 's-', linewidth=2, markersize=6, 
         alpha=0.7, label=f'{N}点滑动平均', color='orange')
plt.xlabel('学生编号', fontsize=12)
plt.ylabel('成绩（分）', fontsize=12)
plt.title('滑动平均滤波后的成绩', fontsize=13, fontweight='bold')
plt.grid(True, alpha=0.3, linestyle='--')
plt.axhline(y=60, color='r', linestyle='--', alpha=0.5)
plt.legend(fontsize=10)
plt.ylim(0, 105)

plt.tight_layout()
plt.savefig('d:/Code/Signals_and_Systems/hw05/scores_before_after_filtering.png', dpi=300, bbox_inches='tight')
plt.show()

# 对比图（叠加显示）
plt.figure(figsize=(12, 6))
plt.plot(range(1, len(scores)+1), scores, 'o-', linewidth=2, markersize=5, 
         alpha=0.6, label='原始成绩', color='blue')
plt.plot(n_filtered+1, scores_filtered, 's-', linewidth=2.5, markersize=6, 
         alpha=0.8, label=f'{N}点滑动平均', color='red')
plt.xlabel('学生编号', fontsize=12)
plt.ylabel('成绩（分）', fontsize=12)
plt.title('滑动平均滤波前后对比', fontsize=14, fontweight='bold')
plt.grid(True, alpha=0.3, linestyle='--')
plt.axhline(y=60, color='green', linestyle='--', alpha=0.5, label='及格线')
plt.legend(fontsize=11)
plt.ylim(0, 105)
plt.tight_layout()
plt.savefig('d:/Code/Signals_and_Systems/hw05/scores_comparison.png', dpi=300, bbox_inches='tight')
plt.show()

# 滤波前后的统计对比
print("\n滤波前后统计对比:")
print(f"原始数据 - 平均值: {mean_score:.2f}, 标准差: {np.std(scores):.2f}")
print(f"滤波数据 - 平均值: {np.mean(scores_filtered):.2f}, 标准差: {np.std(scores_filtered):.2f}")
print(f"\n说明: 滑动平均滤波器平滑了数据的波动，降低了标准差，使数据更加平滑。")

print("\n" + "=" * 60)
print("所有任务完成！")
print("=" * 60)
