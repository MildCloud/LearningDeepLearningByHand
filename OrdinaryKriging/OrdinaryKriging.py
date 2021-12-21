from pykrige.ok import OrdinaryKriging
import numpy as np
#from matplotlib import pyplot as plt

# 已知采样点的数据，是坐标（x，y）和坐标对应的值
# 矩阵中第一列是x,第二列是y,第三列是坐标对应的值
data = np.array(
    [
        [0.1, 0.1, 0.8],
        [0.2, 0.1, 0.8],
        [0.1, 0.3, 0.33],
        [0.5, 0.4, 0.5],
        [0.3, 0.3, 0.7],
        [0.3, 0.7, 0.5],
    ])

# 网格
x_range = 0.8
y_range = 0.7
range_step = 0.1
gridx = np.arange(0.0, x_range, range_step) #三个参数的意思：范围0.0 - 0.6 ，每隔0.1划分一个网格
gridy = np.arange(0.0, y_range, range_step)

ok3d = OrdinaryKriging(data[:, 0], data[:, 1], data[:, 2], variogram_model="linear") # 模型
# variogram_model是变差函数模型，pykrige提供 linear, power, gaussian, spherical, exponential, hole-effect几种variogram_model可供选择，默认的为linear模型。
# 使用不同的variogram_model，预测效果是不一样的，应该针对自己的任务选择合适的variogram_model。

k3d1, ss3d = ok3d.execute("grid", gridx, gridy) # k3d1是结果，给出了每个网格点处对应的值

#print(ss3d)
print(np.round(k3d1,2))