import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
# mnist = input_data.read_data_sets('MNIST_data/', one_hot=True)
mnist = input_data.read_data_sets('C:/Users/NANA/Anaconda3/envs/TensorFlow-cpu/Lib/site-packages/tensorflow/contrib/learn/python/learn/datasets/', one_hot=True)
sess = tf.InteractiveSession()     #建立一个新的session
x = tf.placeholder(tf.float32, [None, 784])
# 用于输入数据，第一个参数为数据类型， 第二个参数 代表tensor 的shape，即数据尺寸，
# None代表不限制条数的输入， 784 代表每条输入为784维的向量
W = tf.Variable(tf.zeros([784, 10]))   #w = [784, 10]
b = tf.Variable(tf.zeros([10]))
#  b [10,]
y = tf.nn.softmax(tf.matmul(x, W) + b)
#定义loss
y_ = tf.placeholder(tf.float32, [None, 10])
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)
tf.global_variables_initializer().run()
for i in range(1000):
    batch_xs, batch_ys = mnist.train.next_batch(100)
    train_step.run({x: batch_xs, y_: batch_ys})


####  testing
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
print(accuracy.eval({x: mnist.test.images, y_: mnist.test.labels}))
