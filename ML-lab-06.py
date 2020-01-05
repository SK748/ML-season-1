import tensorflow as tf
import numpy as np

xy = np.loadtxt('data-04-zoo.csv',delimiter=',', dtype = np.float32)
x_data = xy[:,0:-1]
y_data = xy[:,[-1]]
# nb_clases : w 변수의 개수를 정한다고 생각 -> y가 0~6까지 존재 즉 w1,w2, ... 이 총 7개 존재
nb_classes = 7

X = tf.placeholder(tf.float32, shape =[None,16])
Y = tf.placeholder(tf.int32, shape =[None,1])

# Y에 들어온 0~6까지의 값들을 tf.one_hot을 통해 우리가 원하는 one_hot으로 바꿈
Y_one_hot = tf.one_hot(Y, nb_classes) # one hot shpae = (?, 1, 7)

# one_hot은 차원을 한단계 올려주는 계산 -> one_hot을 하고 차원을 낮춰야함 ->reshape 이용
Y_one_hot = tf.reshape(Y_one_hot, [-1,nb_classes]) # shape = (?,7) -> -1을 넣어 이용 ( -1 : everything)

# X*W + b = Y 가 되어야 하기 때문에 -> None x 16 * 16 x 7 -> None x 7 이 되어야 함
W = tf.Variable(tf.random_normal([16, nb_classes]), name = 'weight')
b = tf.Variable(tf.random_normal([nb_classes]), name = 'bias')

logits = tf.matmul(X,W) + b
hypothesis = tf.nn.softmax(logits)

cost_i = tf.nn.softmax_cross_entropy_with_logits(logits = logits, labels = Y_one_hot)

cost = tf.reduce_mean(cost_i)
optimizer = tf.train.GradientDescentOptimizer(learning_rate = 0.1).minimize(cost)

# prediction & accuracy
# tf.argmax : 가장 큰 값의 index번호를 반환하는 함수( 0:열 ,1:행)
prediction = tf.argmax(hypothesis, 1)
correct_prediction = tf.equal(prediction, tf.argmax(Y_one_hot, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

sess = tf.Session()
sess.run(tf.global_variables_initializer())

for step in range(2000):
    sess.run(optimizer,feed_dict = {X:x_data, Y:y_data})
    if step % 100 ==0:
        loss, acc = sess.run([cost,accuracy],feed_dict = {X:x_data, Y:y_data})
        print("Step : {:5}\t Loss : {:.3f}\t Acc : {:.2%}".format(step, loss, acc))
        
# we can predict
pred = sess.run(prediction, feed_dict = {X:x_data})
for p,y in zip(pred, y_data.flatten()):
    print("[{}] Prediction : {} True Y : {}".format(p == int(y), p, int(y)))