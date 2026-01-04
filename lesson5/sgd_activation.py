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
b =0.1
z =w*xs +b
a=1/(1+np.exp(-z))
plt.plot(xs,a)
plt.show()

for _ in range(5000):
    for i in range(100):
        x=xs[i]
        y=ys[i]
        #对w和b求(偏)导
        z = w*x+b
        a=1/(1+np.exp(-z))
        e =(y-a)**2
        deda =-2*(y-a)
        dadz = a*(1-a)
        dzdw =x
        dedw = deda*dadz*dzdw
        dzdb =1
        dedb = deda*dadz*dzdb
        alpha = 0.1
        w = w - alpha*dedw
        b = b - alpha*dedb
    if _ % 100 == 0 :
        plt.clf()#清空窗口
        plt.scatter(xs,ys)
        z = w*xs+b
        a=1/(1+np.exp(-z))
        plt.xlim(0,1)
        plt.ylim(-0.2,1.2)
        plt.plot(xs,a) #这里xs,a都是向量，用来画线。如果是标量xs[i]和a画的就是点，看不到东西
        plt.pause(0.001)#暂停0.01s