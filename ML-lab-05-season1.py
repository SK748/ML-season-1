import tensorflow as tf

x_data = [[1,2],[2,3],[3,1],[4,3],[5,3],[6,2]]
y_data = [[0],[0],[0],[1],[1],[1]]

X = tf.placeholder(tf.float32,shape = [None,2])
Y = tf.placeholder(tf.float32,shape = [None,1])

W = tf.Variable(tf.random_normal([2,1]),name = 'weight')
b = tf.Variable(tf.random_normal([1]),name = 'bias')

# Logistic classification
hypothesis = tf.sigmoid(tf.matmul(X,W) + b)

# Logistic cost function
cost = -tf.reduce_mean(Y*tf.log(hypothesis) + (1-Y)*tf.log(1 - hypothesis))

# Minimize : Gradient Descent using derivative
# learing_rate의 값이 cost의 결정에 영향을 주기 때문에 적절한 값을 찾아야 함 
optimizer = tf.train.GradientDescentOptimizer(learning_rate = 0.1)
train = optimizer.minimize(cost)

# tf.cast : casting, float -> int , bool : true -> 1, false -> 0
# tf.equal : same -> true, different -> false
# 즉 predicted 에서 hypothesis의 값이 0.5보다 크면 true, 작으면 false -> tf.cast(true) = 1
# tf.cast(false) = 0 이 됨
predicted = tf.cast(hypothesis > 0.5, dtype = tf.float32)
# tf.equal에서 predicted와 Y의 값을 비교해 같으면 true, 다르면 false
# 이후 true와 false의 값들의 평균을 내기 위해 숫자로 변경해줌(cast)
# 평균을 내서 accuracy 확인
accuracy = tf.reduce_mean(tf.cast(tf.equal(predicted, Y), dtype = tf.float32))


sess = tf.Session()

sess.run(tf.global_variables_initializer())

for step in range(2001):
    cost_val,_ = sess.run([cost,train],feed_dict={X:x_data, Y:y_data})
    if step % 200 ==0:
        print(step, cost_val)
        
'''print("hypothesis\n",sess.run(hypothesis,feed_dict={X:x_data, Y:y_data}) ,"correct",sess.run(predicted,feed_dict={X:x_data, Y:y_data}) ,"accurcay",sess.run(accuracy,feed_dict={X:x_data, Y:y_data}))'''