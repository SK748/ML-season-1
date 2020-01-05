import tensorflow as tf
import numpy as np

x_data = np.array([[0,0],[0,1],[1,0],[0,0]])
y_data = np.array([[1],[0],[0],[1]])

X = tf.placeholder(tf.float32)
Y = tf.placeholder(tf.float32)


# 기존 layer를 1개에서 4개로 늘림 , wide를 2 -> 10으로 늘려 봄
W1 = tf.Variable(tf.random_normal([2,10]),name = 'weight')
b1 = tf.Variable(tf.random_normal([10]),name = 'bias')
layer1 = tf.sigmoid(tf.matmul(X,W1) + b1)

# W1의 output이 W2의 input에 영향 
W2 = tf.Variable(tf.random_normal([10,10]),name = 'weight')
b2 = tf.Variable(tf.random_normal([10]),name = 'bias')
layer2 = tf.sigmoid(tf.matmul(layer1,W2) + b2)

# W2의 output이 W3의 input에 영향
W3 = tf.Variable(tf.random_normal([10,10]),name = 'weight')
b3 = tf.Variable(tf.random_normal([10]),name = 'bias')
layer3 = tf.sigmoid(tf.matmul(layer2,W3) + b3)

# W3의 output이 W4의 input에 영향 , W4의 output은 Y에 영향을 받음
W4 = tf.Variable(tf.random_normal([10,1]),name = 'weight')
b4 = tf.Variable(tf.random_normal([1]),name = 'bias')
hypothesis = tf.sigmoid(tf.matmul(layer3,W4) + b4)

# logistic regression
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
    
# 08-3 보다 높은 정확도