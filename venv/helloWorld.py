from tensorflow.examples.tutorials.mnist import input_data
# mnist = input_data.read_data_sets('MNIST_data/', one_hot=True)
mnist = input_data.read_data_sets('C:/Users/NANA/Anaconda3/envs/TensorFlow-cpu/Lib/site-packages/tensorflow/contrib/learn/python/learn/datasets/', one_hot=True)

print(mnist.train.images.shape, mnist.train.labels.shape)
print(mnist.test.images.shape, mnist.test.labels.shape)
print(mnist.validation.images.shape, mnist.validation.labels.shape)