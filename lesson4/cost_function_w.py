import dataset
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

xs,ys = dataset.get_beans("dataset_2.pt")

#配置图像
plt.title("size-Toxicity Function",fontsize=12)#设置图像名称
plt.xlabel("Bean size")# #设置横坐标的名字
plt.ylabel("Toxicity")# #设置纵坐标的名字
plt.xlim(0,1)
plt.ylim(0,1.5)
plt.scatter(xs,ys)
w=0.1
b=0.1
y_pre = w*xs + b
plt.plot(xs,y_pre)
plt.show()
fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
ax = Axes3D(fig)
ax.set_zlim(0,2)

ws = np.arange(-1,2,0.1)
bs = np.arange(-2,2,0.1)
for b in bs:
    es = []
    for w in ws:
        y_pre =w*xs + b
        e =np.sum((ys-y_pre)**2)*(1/100)
        es.append(e)
    ax.plot(ws, es, b, zdir="y")
plt.show()