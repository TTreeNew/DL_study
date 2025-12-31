import torch
import numpy as np
import matplotlib.pyplot as plt

def get_beans(dataset_path = "dataset_1.pt"):
    X, y = torch.load(dataset_path)
    # 可视化数据点
    plt.scatter(X, y, c='blue', label='Data points')  # 所有点用同一种颜色
    plt.title('Beans Data points')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    plt.show()

def main():
    get_beans()

if __name__ == '__main__':
    main()
