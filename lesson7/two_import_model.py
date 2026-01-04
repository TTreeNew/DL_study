import dataset
import numpy as np
import matplotlib.pyplot as plt

xs,ys,zs = dataset.get_beans("dataset_5.pt")

#配置图像
plt.title("size-color-Toxicity Function",fontsize=12)#设置图像名称
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(xs,ys,zs) # 画散点
ax.set_xlabel('Bean Size')
ax.set_ylabel('Color depth')
ax.set_zlabel('Toxicity')
ax.set_xlim(0,1)
ax.set_ylim(0,1)
ax.set_zlim(0,1)

def sigmoid(x):
    return 1/(1+np.exp(-x))
#共一层神经元
x1 = xs
x2 = ys
w1 =np.random.rand()
w2 =np.random.rand()
b = np.random.rand()

#前向传播
def forward_propgation(x1,x2):
    z = x1*w1+x2*w2+b
    a = sigmoid(z)
    return a,z
a,z = forward_propgation(x1,x2)

# 网格# 画预测面
x = np.linspace(0, 1, 100)
y = np.linspace(0, 1, 100)
X, Y = np.meshgrid(x, y)
ax.plot_surface(X, Y, z, alpha=0.5, color='orange')
ax.legend()
plt.show()

# for _ in range(10000):
#     for i in range(200):
#         x=xs[i]
#         y=ys[i]
#         #先前向传播
#         a1_2,z1_2,a2_1,z2_1,a1_1,z1_1 = forward_propgation(x)
#         #反向传播
#         #误差代价e
#         e = (y-a1_2)**2
#         deda1_2 = -2*(y-a1_2)

#         da1_2dz1_2=a1_2*(1-a1_2)

#         dz1_2dw11_2 = a1_1
#         dz1_2dw21_2 = a2_1

#         dedw11_2 = deda1_2*da1_2dz1_2*dz1_2dw11_2
#         dedw21_2 = deda1_2*da1_2dz1_2*dz1_2dw21_2

#         dz1_2db1_2 = 1
#         dedb1_2 = deda1_2*da1_2dz1_2*dz1_2db1_2

#         dz1_2da1_1=w11_2
#         da1_1dz1_1= a1_1*(1-a1_1)
#         dz1_1dw11_1=x
#         dedw11_1 = deda1_2*da1_2dz1_2*dz1_2da1_1*da1_1dz1_1*dz1_1dw11_1

#         dz1_1db1_1=1
#         dedb1_1 = deda1_2*da1_2dz1_2*dz1_2da1_1 *da1_1dz1_1*dz1_1db1_1

#         dz1_2da2_1= w21_2
#         da2_1dz2_1=a2_1*(1-a2_1)
#         dz2_1dw12_1=x
#         dedw12_1 = deda1_2*da1_2dz1_2*dz1_2da2_1*da2_1dz2_1*dz2_1dw12_1

#         dz2_1db2_1=1
#         dedb2_1 = deda1_2*da1_2dz1_2*dz1_2da2_1*da2_1dz2_1*dz2_1db2_1

#         alpha = 0.03 #更新权重和偏置项
#         w11_2= w11_2 - alpha*dedw11_2
#         w21_2= w21_2 - alpha*dedw21_2
#         b1_2 =b1_2 - alpha*dedb1_2

#         w11_1=w11_1 - alpha*dedw11_1
#         b1_1=b1_1 - alpha*dedb1_1

#         w12_1=w12_1- alpha*dedw12_1
#         b2_1 = b2_1 - alpha*dedb2_1


#     if _ % 100 == 0 :
#         plt.clf()#清空窗口
#         plt.scatter(xs,ys)
#         a1_2,z1_2,a2_1,z2_1,a1_1,z1_1 = forward_propgation(xs)
#         plt.xlim(0,1)
#         plt.ylim(-0.2,1.2)
#         plt.plot(xs,a1_2) #这里xs,a都是向量，用来画线。如果是标量xs[i]和a画的就是点，看不到东西
#         plt.pause(0.001)#暂停0.01s