import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
# mnist = input_data.read_data_sets('MNIST_data/', one_hot=True)
mnist = input_data.read_data_sets('C:/Users/NANA/Anaconda3/envs/TensorFlow-cpu/Lib/site-packages/tensorflow/contrib/learn/python/learn/datasets/', one_hot=True)
sess = tf.InteractiveSession()     #建立一个新的session
in_units = 784
h1_units = 300
W1 = tf.Variable(tf.truncated_normal([in_units, h1_units], stddev = 0.1))
b1  = tf.Variable(tf.truncated_normal([h1_units]))
#隐含层各个元素
W2 = tf.Variable(tf.zeros([h1_units, 10]))
b2 = tf.Variable(tf.zeros([10]))
x = tf.placeholder(tf.float32, [None, in_units])
keep_prob = tf.placeholder(tf.float32)
# 存放 即为Dropout 概率   保留节点的比率  训练时<1, 预测时=1
hidden1 = tf.nn.relu(tf.matmul(x, W1) + b1)
# 定义隐含层 y = relu(W1 * x + b)
hidden1_drop = tf.nn.dropout(hidden1, keep_prob)
y = tf.nn.softmax(tf.matmul(hidden1_drop, W2) + b2)
y_ = tf.placeholder(tf.float32, [None, 10])
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))
train_step = tf.train.AdagradOptimizer(0.3).minimize(cross_entropy)
tf.global_variables_initializer().run()
for i in range(1000):
    batch_xs, batch_ys = mnist.train.next_batch(100)
    train_step.run({x: batch_xs, y_: batch_ys, keep_prob : 0.75})



correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
print(accuracy.eval({x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}))


