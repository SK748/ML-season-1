# mnist 문제에서 cnn을 3번, FC를 2번 한 것
import tensorflow as tf
import random
import matplotlib.pyplot as plt

from tensorflow.examples.tutorials.mnist import input_data

tf.set_random_seed(777)

mnist = input_data.read_data_sets("MNIST_data/", one_hot = True)

learning_rate = 0.001
training_epochs = 15
batch_size = 100

class Model:
    
    def __init__(self, sess, name):
        self.sess = sess
        self.name = name
        self.__build_net()

    def __build_net(self):
        with tf.variable_scope(self.name):
            
            self.keep_prob = tf.placeholder(tf.float32)
            self.X = tf.placeholder(tf.float32, [None, 784])
            X_img = tf.reshape(self.X, [-1, 28, 28, 1])
            self.Y = tf.placeholder(tf.float32, [None, 10])

            # L1 filter
            W1 = tf.Variable(tf.random_normal([3, 3, 1, 32], stddev = 0.01))
            # L1 imgln shape = (?, 28, 28, 1)
            # L1 conv shape = (?, 28, 28, 32)
            # L1 pool shape = (?, 14, 14, 32)
            L1 = tf.nn.conv2d(X_img, W1, strides = [1, 1, 1, 1], padding = 'SAME')
            L1 = tf.nn.relu(L1)
            L1 = tf.nn.max_pool(L1, ksize = [1, 2, 2, 1], strides = [1, 2, 2, 1], padding = 'SAME')
            L1 = tf.nn.dropout(L1, keep_prob = self.keep_prob)
    
            W2 = tf.Variable(tf.random_normal([3, 3, 32, 64], stddev = 0.01))
            # L2 imgln shape = (?, 14, 14, 32)
            # L2 conv shape = (?, 14, 14, 64)
            # L2 pool shape = (?, 7, 7, 64)
            L2 = tf.nn.conv2d(L1, W2, strides = [1, 1, 1, 1], padding = 'SAME')
            L2 = tf.nn.relu(L2)
            L2 = tf.nn.max_pool(L2, ksize = [1, 2, 2, 1], strides = [1, 2, 2, 1], padding = 'SAME')
            L2 = tf.nn.dropout(L2, keep_prob = self.keep_prob)
    
            W3 = tf.Variable(tf.random_normal([3, 3, 64, 128], stddev = 0.01))
            # L3 imgln shape = (?, 7, 7, 64)
            # L3 conv shape = (?, 7, 7, 128)
            # L3 pool shape = (?, 4, 4, 128)
            L3 = tf.nn.conv2d(L2, W3, strides=[1, 1, 1, 1], padding='SAME')
            L3 = tf.nn.relu(L3)
            L3 = tf.nn.max_pool(L3, ksize = [1, 2, 2, 1], strides =[1, 2, 2, 1], padding = 'SAME')
            L3 = tf.nn.dropout(L3, keep_prob = self.keep_prob)
            L3_flat = tf.reshape(L3, [-1, 128*4*4])
    
            # fc 4*4*128 input -> 625 output
            W4 = tf.get_variable("W10111120", shape = [4*4*128, 625],initializer = tf.contrib.layers.xavier_initializer())
            b4 = tf.Variable(tf.random_normal([625]))
            L4 = tf.nn.relu(tf.matmul(L3_flat, W4) + b4)
            L4 = tf.nn.dropout(L4, keep_prob = self.keep_prob)
    
            # 625 input -> 10 output
            W5 = tf.get_variable("W11211011", shape = [625, 10], initializer = tf.contrib.layers.xavier_initializer())
            b5 = tf.Variable(tf.random_normal([10]))
            self.logits = tf.matmul(L4,W5) +  b5
    
        self.cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = self.logits, labels = self.Y))
        self.optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate).minimize(self.cost)
        correct_prediction = tf.equal(tf.argmax(self.logits, 1), tf.argmax(self.Y, 1))
        self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    def predicted(self, x_test, keep_prob = 0.7):
        return self.sess.run(self, logits, feed_dict={self.X: x_test, self.keep_prob:keep_prob})

    def get_accuracy(self, x_test, y_test, keep_prop=1.0):
        return self.sess.run(self.accuracy, feed_dict={self.X: x_test, self.Y: y_test, self.keep_prob: keep_prop})

    def train(self, x_data, y_data, keep_prop=0.7):
        return self.sess.run([self.cost, self.optimizer], feed_dict={self.X: x_data, self.Y: y_data, self.keep_prob: keep_prop})

# initialize
sess = tf.Session()
m1 = Model(sess, "m1")

sess.run(tf.global_variables_initializer())


print('Learning started. It takes sometime.')
for epoch in range(training_epochs):
    avg_cost = 0
    total_batch = int(mnist.train.num_examples / batch_size)
    for i in range(total_batch):
        batch_xs, batch_ys = mnist.train.next_batch(batch_size)
        c,_ = m1.train(batch_xs, batch_ys)
        avg_cost += c/total_batch

    print('Epoch :','%04d' % (epoch + 1),'cost =', '{:9f}'.format(avg_cost))
print('Learning Finished!')    

        

print('Accuracy:', m1.get_accuracy(mnist.test.images, mnist.test.labels))