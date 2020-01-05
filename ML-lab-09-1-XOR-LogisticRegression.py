import tensorflow as tf
import numpy as np

x_data = np.array([[0,0],[0,1],[1,0],[0,0]])
y_data = np.array([[1],[0],[0],[1]])

X = tf.placeholder(tf.float32)
Y = tf.placeholder(tf.float32)

W = tf.Variable(tf.random_normal([2,1]),name = 'weight')
b = tf.Variable(tf.random_normal([1]),name = 'bias')

# logistic regression
hypothesis = tf.sigmoid(tf.matmul(X,W) + b)
cost = -tf.reduce_mean(Y*tf.log(hypothesis) + (1-Y)*tf.log(1-hypothesis))
train = tf.train.GradientDescentOptimizer(learning_rate = 0.1).minimize(cost)

predicted = tf.cast(hypothesis>0.5, dtype = tf.float32)
accuracy = tf.reduce_mean(tf.cast(tf.equal(predicted,Y),dtype = tf.float32))

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    
    for step in range(10001):
        sess.run(train,feed_dict={X:x_data, Y:y_data})
        if step % 200 == 0:
            print(step, sess.run(cost, feed_dict={X:x_data, Y:y_data}),sess.run(W))

# Accuracy report

    h,c,a = sess.run([hypothesis, predicted, accuracy], feed_dict={X:x_data, Y:y_data})
    print("hypothesis :", h,"Coorect :",c, "Accuracy :", a)
    
# 이 과정상의 문제는 없지만, 원하는 accuracy가 나오지 않을 가능성이 존재 -> 왜냐하면 layer가 하나이기 때문 
# 위 문제를 해결하기 위해 Neural Net을 통해 정확도를 높이자 (2가지 방법 존재 : 1. wide를 넓히기, 2. layers의 개수 늘리기)