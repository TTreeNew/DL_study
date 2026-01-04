import dataset
import numpy as np
import matplotlib.pyplot as plt

xs,ys = dataset.get_beans("dataset_4.pt")

#配置图像
plt.title("size-Toxicity Function",fontsize=12)#设置图像名称
plt.xlabel("Bean size")# #设置横坐标的名字
plt.ylabel("Toxicity")# #设置纵坐标的名字
plt.scatter(xs,ys)

def sigmoid(x):
    return 1/(1+np.exp(-x))
#第一层
#第一个神经元
w11_1 =np.random.rand()
b1_1 = np.random.rand()
#二个神经元
w12_1 =np.random.rand()
b2_1 = np.random.rand()
#第二层
w11_2 = np.random.rand()
w21_2 = np.random.rand()
b1_2 = np.random.rand()
#前向传播
def forward_propgation(xs):
    z1_1 = w11_1*xs + b1_1
    a1_1 =sigmoid(z1_1)

    z2_1 = w12_1*xs + b2_1
    a2_1 = sigmoid(z2_1)

    z1_2 = w11_2*a1_1 + w21_2*a2_1 + b1_2
    a1_2 = sigmoid(z1_2)
    return a1_2,z1_2,a2_1,z2_1,a1_1,z1_1
a1_2,z1_2,a2_1,z2_1,a1_1,z1_1 = forward_propgation(xs)
plt.plot(xs,a1_2)
plt.show()

for _ in range(10000):
    for i in range(100):
        x=xs[i]
        y=ys[i]
        #先前向传播
        a1_2,z1_2,a2_1,z2_1,a1_1,z1_1 = forward_propgation(x)
        #反向传播
        #误差代价e
        e = (y-a1_2)**2
        deda1_2 = -2*(y-a1_2)

        da1_2dz1_2=a1_2*(1-a1_2)

        dz1_2dw11_2 = a1_1
        dz1_2dw21_2 = a2_1

        dedw11_2 = deda1_2*da1_2dz1_2*dz1_2dw11_2
        dedw21_2 = deda1_2*da1_2dz1_2*dz1_2dw21_2

        dz1_2db1_2 = 1
        dedb1_2 = deda1_2*da1_2dz1_2*dz1_2db1_2

        dz1_2da1_1=w11_2
        da1_1dz1_1= a1_1*(1-a1_1)
        dz1_1dw11_1=x
        dedw11_1 = deda1_2*da1_2dz1_2*dz1_2da1_1*da1_1dz1_1*dz1_1dw11_1

        dz1_1db1_1=1
        dedb1_1 = deda1_2*da1_2dz1_2*dz1_2da1_1 *da1_1dz1_1*dz1_1db1_1

        dz1_2da2_1= w21_2
        da2_1dz2_1=a2_1*(1-a2_1)
        dz2_1dw12_1=x
        dedw12_1 = deda1_2*da1_2dz1_2*dz1_2da2_1*da2_1dz2_1*dz2_1dw12_1

        dz2_1db2_1=1
        dedb2_1 = deda1_2*da1_2dz1_2*dz1_2da2_1*da2_1dz2_1*dz2_1db2_1

        alpha = 0.03 #更新权重和偏置项
        w11_2= w11_2 - alpha*dedw11_2
        w21_2= w21_2 - alpha*dedw21_2
        b1_2 =b1_2 - alpha*dedb1_2

        w11_1=w11_1 - alpha*dedw11_1
        b1_1=b1_1 - alpha*dedb1_1

        w12_1=w12_1- alpha*dedw12_1
        b2_1 = b2_1 - alpha*dedb2_1


    if _ % 100 == 0 :
        plt.clf()#清空窗口
        plt.scatter(xs,ys)
        a1_2,z1_2,a2_1,z2_1,a1_1,z1_1 = forward_propgation(xs)
        plt.xlim(0,1)
        plt.ylim(-0.2,1.2)
        plt.plot(xs,a1_2) #这里xs,a都是向量，用来画线。如果是标量xs[i]和a画的就是点，看不到东西
        plt.pause(0.001)#暂停0.01s