import numpy as np
import matplotlib.pyplot as plt

t = np.linspace(0, 10, 1000)  # 时间范围0~10，取1000个点
r = 2*np.exp(-t) - (5/2)*np.exp(-2*t) + 3/2  # 全响应表达式

plt.plot(t, r)
plt.xlabel('t')
plt.ylabel('r(t)')
plt.title('System Response')
plt.grid(True)  # 显示网格
plt.show()
