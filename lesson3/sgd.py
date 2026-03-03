#无激活函数无偏置项的1x1神经网络
import dataset
import numpy as np
import matplotlib.pyplot as plt

xs,ys = dataset.get_beans("dataset_3.pt")

#配置图像
plt.title("size-Toxicity Function",fontsize=12)#设置图像名称
plt.xlabel("Bean size")# #设置横坐标的名字
plt.ylabel("Toxicity")# #设置纵坐标的名字
plt.scatter(xs,ys)
w=0.1
y_pre = w*xs
plt.plot(xs,y_pre)
plt.show()

for _ in range(100):
    for i in range(100):
        x= xs[i]
        y = ys[i]
        #y=wx
        #方差(y-wx)^2
        #方差关于权重w的梯度k=2(y-wx)(-x)
        k= 2*(x**2)*w +(-2*x*y)
        alpha = 0.1
        w= w - alpha*k
        plt.clf()#清空窗口
        plt.scatter(xs,ys)
        y_pre = w*xs
        plt.xlim(0,1)
        plt.ylim(0,1.2)
        plt.plot(xs,y_pre)
        plt.pause(0.01)#暂停0.01s 