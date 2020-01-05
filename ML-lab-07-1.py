import tensorflow as tf

# lab-06과 다르게 y_data가 one_hot 상태
x_data = [[1,2,1],[1,3,2],[1,3,4],[1,5,5],[1,7,5],[1,2,5],[1,6,6],[1,7,7]]
y_data = [[0,0,1],[0,0,1],[0,0,1],[0,1,0],[0,1,0],[0,1,0],[1,0,0],[1,0,0]]

x_test = [[2,1,1],[3,1,2],[3,3,4]]
y_test = [[0,0,1],[0,0,1],[0,0,1]]

X = tf.placeholder("float", shape = [None,3])
Y = tf.placeholder("float", shape = [None,3])

W = tf.Variable(tf.random_normal([3,3]),name = 'weight')
b = tf.Variable(tf.random_normal([3]),name = 'bias')

hypothesis = tf.nn.softmax(tf.matmul(X,W) + b)
cost = tf.reduce_mean(-tf.reduce_sum(Y*tf.log(hypothesis), axis = 1))
optimizer = tf.train.GradientDescentOptimizer(learning_rate = 0.1).minimize(cost)

prediction = tf.argmax(hypothesis,1)
# tf.argmax(Y,1)자리에 y_data도 된다고 생각했음 -> y_data를 넣으면 기존 data를 학습시키는 과정에서는 문제가 없지만, 
# y_test를 불러와 accuracy를 확인 할 때 y_test의 변수가 is_correct에 들어가지 않음 -> accuracy가 다르게 나옴
is_correct = tf.equal(prediction, tf.argmax(Y,1))
accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for step in range(201):
        cost_val, W_val, _ = sess.run([cost,W,optimizer],feed_dict = {X:x_data,Y:y_data})
        if step % 40 ==0:
            print(step, cost_val, W_val)

    print("Prediction :", sess.run(prediction, feed_dict = {X:x_test}))

    print("Accuracy : ", sess.run(accuracy, feed_dict = {X:x_test, Y:y_test}))