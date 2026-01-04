import torch
import matplotlib.pyplot as plt

# 生成随机数据点
n = 200
x = torch.rand(n)
y = torch.rand(n)
# x, _ = torch.sort(x)
# y, _ = torch.sort(y)
z = torch.zeros(n)

# 生成圆形毒区
mask = y > 0.6 #mask为保存布尔值的一维张量
z[mask] = 1  

# 保存数据
torch.save((x, y, z), 'dataset_5.pt')

# 3D 可视化
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

ax.scatter(x,y,z,label='Data points')


ax.set_xlabel('Size')
ax.set_ylabel('Color depth')
ax.set_zlabel('Toxicity')
ax.set_xlim(0,1)
ax.set_ylim(0,1)
ax.set_zlim(0,1)
ax.legend()
plt.show()
