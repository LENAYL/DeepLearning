import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
# mnist = input_data.read_data_sets('MNIST_data/', one_hot=True)
mnist = input_data.read_data_sets('C:/Users/NANA/Anaconda3/envs/TensorFlow-cpu/Lib/site-packages/tensorflow/contrib/learn/python/learn/datasets/', one_hot=True)
sess = tf.InteractiveSession()     #建立一个新的session

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev = 0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides= [1, 1, 1, 1], padding= 'SAME')

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

x = tf.placeholder(tf.float32, [None, 784])
y_ = tf.placeholder(tf.float32, [None, 10])
x_image = tf.reshape(x, [-1, 28, 28, 1])
# -1 表示输入数据量不固定  28*28 = 1*784 把图像转化为原始的2D 最后的1 代表1通道 因为只有灰度图像 只需要一个通道 如果是RGB则需要3通道
W_conv1 = weight_variable([5, 5, 1, 32])
#32个卷积核
b_conv1 = bias_variable([32])
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)


