# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

# 设置中文字体以避免绘图时出现方块（豆腐块）
# 优先使用 Windows 常见字体 Microsoft YaHei、回退到 SimHei，最后回退到常见 Unicode 字体
mpl.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'Arial Unicode MS']
mpl.rcParams['font.family'] = 'sans-serif'
# 让坐标轴的负号正常显示（否则可能显示为方块）
mpl.rcParams['axes.unicode_minus'] = False

t = np.linspace(0, 10, 1000)  # 时间范围0~10，取1000个点
全响应 = 2*np.exp(-t) - (5/2)*np.exp(-2*t) + 3/2
暂态响应 = 2*np.exp(-t) - (5/2)*np.exp(-2*t)
稳态响应 = 3/2 * np.ones_like(t)  # 生成与t等长的稳态值数组

plt.figure(figsize=(10, 6))
plt.plot(t, 全响应, 'r-', label='全响应')
plt.plot(t, 暂态响应, 'g--', label='暂态响应')
plt.plot(t, 稳态响应, 'b:', label='稳态响应')
plt.xlabel('时间 t')
plt.ylabel('响应 r(t)')
plt.title('系统响应波形')
plt.legend()
plt.grid(True)
plt.show()
