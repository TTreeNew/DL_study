import dataset
import numpy as np
import matplotlib.pyplot as plt

# # 启用交互模式 
# plt.ion() 

xs,ys,zs = dataset.get_beans("dataset_5.pt")
x_axis = np.linspace(0, 1, 30)  
y_axis = np.linspace(0, 1, 30)
x_axis, y_axis = np.meshgrid(x_axis, y_axis)  # 生成网格
#配置图像
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
plt.title("size-color-Toxicity Function",fontsize=12)#设置图像名称
ax.scatter(xs,ys,zs) # 画散点
ax.set_xlabel('Bean Size')
ax.set_ylabel('Color depth')
ax.set_zlabel('Toxicity')
ax.set_xlim(0,1)
ax.set_ylim(-0.2,1.2)
ax.set_zlim(-0.2,1.2)

def sigmoid(x):
    return 1/(1+np.exp(-x))
#共一层神经元
w1 =np.random.rand()
w2 =np.random.rand()
b = np.random.rand()

#前向传播
def forward_propgation(x1,x2):
    z = x1*w1+x2*w2+b
    a = sigmoid(z)
    return a,z
a,z = forward_propgation(x_axis,y_axis)
ax.plot_surface(x_axis, y_axis, z, alpha=0.5, color='orange') #画曲面
plt.show()  
# plt.show(block=False)  # 使用 block=False 避免阻塞
# plt.pause(0.1) 

# # 创建新的图形窗口用于训练
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')

for epoch in range(1000):
    for i in range(200):
        x1=xs[i]
        x2=ys[i]
        z1=zs[i]
        #先前向传播
        a,z = forward_propgation(x1,x2)
        #反向传播
        #误差代价e
        e = (z1 - a)**2
        deda = -2*(z1 - a)

        dadz=a*(1-a)

        dzdw1 = x1
        dzdw2 = x2

        dedw1 = deda*dadz*dzdw1
        dedw2 = deda*dadz*dzdw2

        dzdb = 1
        dedb = deda*dadz*dzdb

        alpha = 0.03 #更新权重和偏置项
        w1= w1 - alpha*dedw1
        w2= w2 - alpha*dedw2
        b =b - alpha*dedb


    if epoch % 10 == 0 :
        a,z = forward_propgation(x_axis,y_axis)
        ax.cla()# 清除轴并重新绘制

        # 绘制数据点和曲面
        ax.scatter(xs, ys, zs)
        ax.plot_surface(x_axis,y_axis,a) #这里xs,ys,a都是网格（矩阵），用来画曲面。如果是向量画的就是空间曲线
        # plt.draw()  
        plt.pause(0.01) 


        # ax.collections.clear()#保留窗口清除曲面
        # ax.scatter(xs,ys,zs)
        # a,z = forward_propgation(x_axis,y_axis)
        # # ax.set_xlim(0,1)
        # # ax.set_ylim(-0.2,1.2)
        # # ax.set_zlim(-0.2,1.2)
        # # ax.set_xlabel('Bean Size')
        # # ax.set_ylabel('Color depth')
        # # ax.set_zlabel('Toxicity')
        # ax.plot_surface(x_axis,y_axis,a) #这里xs,ys,a都是网格（矩阵），用来画曲面。如果是向量画的就是空间曲线
        # fig.canvas.draw()
        # plt.pause(0.001)#暂停0.001s