import dataset
import numpy as np
import matplotlib.pyplot as plt

xs,ys = dataset.get_beans("dataset_1.pt")

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
        #a=X^2
        #b=-2*x*y
        #c=y^2
        #斜率k=2aw+b
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