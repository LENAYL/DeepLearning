import tensorflow as tf
import os
import time
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
# start = time.time()
# a = tf.random_normal((100, 100))
# b = tf.random_normal((100, 500))
# c = tf.matmul(a, b)
# sess = tf.InteractiveSession()
# end = time.time()
# print((end - start) * 1000, end='ms')
import numpy as np
import matplotlib.pyplot as plt


#  定义激活函数
# def relu(z):
#     if z < 0:
#         return 0
#     else:
#         return z
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# 定义前向传播
def forward(X, w1, w2, b1, b2):
    z1 = np.dot(w1, X) + b1  # w1=h*n     X=n*m      z1=h*m
    A1 = sigmoid(z1)            # A1=h*m
    z2 = np.dot(w2, A1) + b2 # w2=1*h   z2=1*m
    A2 = sigmoid(z2)            # A2=1*m
    return z1, z2, A1, A2


# 定义反向传播
def backward(y, X, A2, A1, z1, z2, w2, w1):
    n, m = np.shape(X)     # n行 m列
    dz2 = A2 - y           #  输出与预期差  A2=1*m y=1*m
    dw2 = 1/m * np.dot(dz2, A1.T)     # dz2=1*m A1.T=m*h(A1转置) dw2=1*h
    db2 = 1 / m * np.sum(dz2, axis=1, keepdims=True)   # axis  定义取到摸个元素所需要的深度，axis = 1，说明 a[][确定][]
    # r, c = np.shape(np.dot(w2.T, dz2))
    # r, c = np.shape(w1)
    # print(r,c)
    dz1 = np.dot(w2.T, dz2) * A1 * (1 - A1)  # w2.T=h*1 dz2=1*m z1=h*m A1=h*m dz1=h*m
    dw1 = 1 / m * np.dot(dz1, X.T)  # z1=h*m X'=m*n dw1=h*n
    db1 = 1 / m * np.sum(dz1, axis=1, keepdims=True)
    return dw1, dw2, db1, db2

# 定义损失函数
def costfunction(A2, y):
    m, n = np.shape(y)
    j = np.sum(y * np.log(A2) + (1 - y) * np.log(1 - A2)) / m
    return -j

# 输入x 拟合数据y
X=np.random.rand(100,200)
n, m = np.shape(X)
y=np.random.rand(1,m)

# 定义各个参数
n_x = n  # size of the input layer
n_y = 1  # size of the output layer
n_h = 5  # size of the hidden layer
w1 = np.random.randn(n_h, n_x) * 0.01  # h*n
b1 = np.zeros((n_h, 1))  # h*1
w2 = np.random.randn(n_y, n_h) * 0.01  # 1*h
b2 = np.zeros((n_y, 1))
alpha = 0.1
number = 10000


for i in range(0, number):
    z1, z2, A1, A2 = forward(X, w1, w2, b1, b2)
    dw1, dw2, db1, db2 = backward(y, X, A2, A1, z2, z1, w2, w1)
    w1 = w1 - alpha * dw1
    w2 = w2 - alpha * dw2
    b1 = b1 - alpha * db1
    b2 = b2 - alpha * db2
    J = costfunction(A2, y)
    if (i % 100 == 0):
        print(i)
    plt.plot(i, J, 'ro')
plt.show()

