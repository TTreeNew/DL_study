import torch
import numpy as np
import matplotlib.pyplot as plt

# 生成随机数据点
n = 100  # 点的数量
x = torch.linspace(0, 1, n)  # x 范围 [0, 1]
y = 0.5 + 0.5*x + 0.03 * torch.randn(n)  # y = x + 噪声

# 保存数据
torch.save((x, y), 'dataset_1.pt')
# 可视化数据点
plt.scatter(x.numpy(), y.numpy(), c='blue', label='Data points')  # 所有点用同一种颜色
plt.title('Beans Data points')
plt.xlabel('x')
plt.ylabel('y')
plt.xlim(0,1)
plt.ylim(0,1.2)
plt.legend()
plt.show()