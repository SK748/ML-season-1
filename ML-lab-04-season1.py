import tensorflow as tf

x_data = [[73.,80.,75.],[93.,88.,93.],[89.,91.,90.],[96.,98.,100.],[73.,66.,70.]]
y_data = [[152.],[185.],[180.],[196.],[142.]]

W = tf.Variable(tf.random_normal([3,1]),name = 'weight')
b = tf.Variable(tf.random_normal([1]),name = 'bias')

X = tf.placeholder(tf.float32,shape = [None,3])
Y = tf.placeholder(tf.float32,shape = [None,1])

hypothesis = tf.matmul(X,W) + b

cost = tf.reduce_mean(tf.square(hypothesis - Y))

#Minimize : Gradient Descent using derivative
#learing_rate의 값이 cost의 결정에 영향을 주기 때문에 적절한 값을 찾아야 함 
optimizer = tf.train.GradientDescentOptimizer(learning_rate = 1e-5)
train = optimizer.minimize(cost)


sess = tf.Session()

sess.run(tf.global_variables_initializer())

for step in range(2001):
    cost_val, hy_val, _ = sess.run([cost,hypothesis,train],feed_dict={X:x_data,Y:y_data})
    if step % 40 ==0:
        print(step, "Cost :", cost_val, "\nPrediction :\n", hy_val)
        
'''결과 확인 : print(sess.run(hypothesis,feed_dict={X:[[90., 80., 100.]]}))'''

'''데이터 읽는 법 : np.loattxt('데이터명',delimiter=',',dtype=np.float32)'''