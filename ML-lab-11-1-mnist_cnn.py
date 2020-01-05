# mnist 문제에서 cnn을 2번 한 것

import tensorflow as tf
import random
import matplotlib.pyplot as plt

tf.set_random_seed(777)

from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("MNIST_data/",one_hot = True)

learning_rate = 0.001
training_epochs = 15
batch_size = 100

# 0~9까지로 나타내기 때문
nb_classes = 10

X = tf.placeholder(tf.float32,[None, 784])
# image를 28x28x1로 reshape하고, -1은 n개의 이미지를 하는것을 의미
X_img = tf.reshape(X, [-1, 28, 28, 1]) 
Y = tf.placeholder(tf.float32,[None, nb_classes])

# 현재 shape = (?, 28, 28, 1)인 상태
# W1 = ([3, 3, 1, 32]) -> 3x3x1 filter를 32개 사용함, 표준편차 = 0.01
W1 = tf.Variable(tf.random_normal([3, 3, 1, 32], stddev = 0.01))
# Conv shape를 결정 할 L1에서 stride 1x1선택, padding = 'SAME'
# padding = 'SAME' -> stride에 따라 padding을 잘 조절힘 (the ouput may be same or smaller than the input depending on the stride option)
# 따라서 L1 conv shape = (?, 28, 28, 32)
L1 = tf.nn.conv2d(X_img, W1, strides = [1, 1, 1, 1], padding = 'SAME')
L1 = tf.nn.relu(L1)
# max pooling 사용, ksize : filterfh 생각 2x2, stride : 2x2, padding = 'SAME'
# L1 pool shape -> (?, 14, 14, 32)
L1 = tf.nn.max_pool(L1, ksize = [1, 2, 2, 1], strides = [1, 2, 2, 1], padding = 'SAME')
# L1 = tf.nn.dropout(L1, keep_prob) : dropout에 추가해도 되고 안해도 되고, 하면 오를 가능성 존재

# L2 Imgln shape = (?, 14, 14, 32) , L1에서 depth가 32였기 때문에 w2의 filter도 32가 되어야함
W2 = tf.Variable(tf.random_normal([3, 3, 32, 64], stddev = 0.01))

# L2 conv shape = (?, 14, 14, 64)
# L2 pool shape = (?, 7, 7, 64)
L2 = tf.nn.conv2d(L1, W2, strides = [1, 1, 1, 1], padding = 'SAME')
L2 = tf.nn.relu(L2)
L2 = tf.nn.max_pool(L2, ksize = [1, 2, 2, 1], strides = [1, 2, 2, 1], padding = 'SAME')
# L2 = tf.nn.dropout(L2, keep_prob)
# conv, pooling 이 끝나 이 값을 FC에 대입하기위해 reshape를 해야함
# 기존 neural network 에서 X의 모양을 [None, 변수 개수]로 생각하면 됨
# L2_flat shape = (?, 3136)
L2_flat = tf.reshape(L2, [-1, 7*7*64])

# Final FC 7x7x64 inputs -> 10 outputs
W3 = tf.get_variable("W1000", shape = [7*7*64, 10], initializer = tf.contrib.layers.xavier_initializer())

b = tf.Variable(tf.random_normal([10]))
logits = tf.matmul(L2_flat, W3) + b

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = logits, labels = Y))
optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate).minimize(cost)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

print("Learning started. It takes sometime...")
for epoch in range(training_epochs):
    # cost의 평균 초기화
    avg_cost = 0
    total_batch = int(mnist.train.num_examples / batch_size)
    for i in range(total_batch):
        batch_xs, batch_ys = mnist.train.next_batch(batch_size)
        feed_dict = {X: batch_xs, Y: batch_ys}
        c, _ = sess.run([cost,optimizer],feed_dict = feed_dict)
        avg_cost += c/total_batch
    print('Epoch :', '%04d' % (epoch + 1), 'cost =', '{:.9f}'.format(avg_cost))
    
print('Learning Finished!')

# Test model and check accuracy
prediction = tf.equal(tf.argmax(logits, 1),tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(prediction, tf.float32))
print('Accuracy :', sess.run(accuracy, feed_dict={X: mnist.test.images, Y: mnist.test.labels}))

# Get one and predict
r = random.randint(0, mnist.test.num_examples -1)
# 왜 argmax 사용? -> 최초 mnist 받을 때 one_hot 상태로 받음, 즉 테스트 데이터가 1,2,3 과 같은 숫자가아니고 one_hot 형식으로 있기 때문에 tf.argmax를 이용해 index번호를 받아와야함
print("Label : " , sess.run(tf.argmax(mnist.test.labels[r : r + 1], 1)))
    
print("Prediction :", sess.run(tf.argmax(logits, 1), feed_dict = {X : mnist.test.images[r : r + 1]}))

plt.imshow(mnist.test.images[r:r + 1].reshape(28, 28), cmap='Greys', interpolation='nearest')
plt.show()
        
        
        





