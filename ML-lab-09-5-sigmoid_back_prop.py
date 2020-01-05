import tensorflow as tf
import numpy as np
tf.set_random_seed(777)  # for reproducibility

# Predicting animal type based on various features
xy = np.loadtxt('data-04-zoo.csv', delimiter=',', dtype=np.float32)
X_data = xy[:, 0:-1]
N = X_data.shape[0]
y_data = xy[:, [-1]]

# y_data has labels from 0 ~ 6
print("y has one of the following values")
print(np.unique(y_data))

# X_data.shape = (101, 16) => 101 samples, 16 features
# y_data.shape = (101, 1)  => 101 samples, 1 label
print("Shape of X data: ", X_data.shape)
print("Shape of y data: ", y_data.shape)

nb_classes = 7  # 0 ~ 6

X = tf.placeholder(tf.float32, [None, 16])
y = tf.placeholder(tf.int32, [None, 1])  # 0 ~ 6

target = tf.one_hot(y, nb_classes)  # one hot
target = tf.reshape(target, [-1, nb_classes])
target = tf.cast(target, tf.float32)

W = tf.Variable(tf.random_normal([16, nb_classes]), name='weight')
b = tf.Variable(tf.random_normal([nb_classes]), name='bias')


def sigma(x):
    # sigmoid function
    # σ(x) = 1 / (1 + exp(-x))
    return 1. / (1. + tf.exp(-x))


def sigma_prime(x):
    # derivative of the sigmoid function
    # σ'(x) = σ(x) * (1 - σ(x))
    return sigma(x) * (1. - sigma(x))


# Forward propagtion
layer_1 = tf.matmul(X, W) + b
y_pred = sigma(layer_1)

# Loss Function (end of forwad propagation)
loss_i = - target * tf.log(y_pred) - (1. - target) * tf.log(1. - y_pred)
loss = tf.reduce_sum(loss_i)

# Dimension Check
# assert : print로 출력해 확인하는 것이 아닌, assert를 이용하면 아무 일 없으면 그냥 넘어가고, 문제가 발생하면 오류가 발생한다.
assert y_pred.shape.as_list() == target.shape.as_list()


# Back prop (chain rule)
# How to derive? please read "Neural Net Backprop in one slide!"
# loss에 대한 미분값
d_loss = (y_pred - target) / (y_pred * (1. - y_pred) + 1e-7) 
# sigma에 대한 미분값
d_sigma = sigma_prime(layer_1)
# 두개를 곱해(chain rule) 영향을 끼치는 정도를 확인
d_layer = d_loss * d_sigma
# chain rule을 적용하면 d_b 에서 1*d_layer이 되므로 d_b = d_layer
d_b = d_layer
# chain rule을 적용하면 d_W (tf.matmul(X,W)를 미분한 것 * d_layer)이 됨
d_W = tf.matmul(tf.transpose(X), d_layer)

# Updating network using gradients
learning_rate = 0.01
train_step = [
    tf.assign(W, W - learning_rate * d_W),
    tf.assign(b, b - learning_rate * tf.reduce_sum(d_b)),
]

# Prediction and Accuracy
# tf.argmax( ~,1)에서 1은 axis를 의미 즉 axis = 1 -> shape이 2이기 때문에 행 방향으로 최댓값의 index를 반환
prediction = tf.argmax(y_pred, 1)
acct_mat = tf.equal(tf.argmax(y_pred, 1), tf.argmax(target, 1))
acct_res = tf.reduce_mean(tf.cast(acct_mat, tf.float32))

# Launch graph
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for step in range(500):
        sess.run(train_step, feed_dict={X: X_data, y: y_data})

        if step % 10 == 0:
            # Within 300 steps, you should see an accuracy of 100%
            step_loss, acc = sess.run([loss, acct_res], feed_dict={
                                      X: X_data, y: y_data})
            print("Step: {:5}\t Loss: {:10.5f}\t Acc: {:.2%}" .format(
                step, step_loss, acc))

    # Let's see if we can predict
    pred = sess.run(prediction, feed_dict={X: X_data})
    for p, y in zip(pred, y_data):
        msg = "[{}]\t Prediction: {:d}\t True y: {:d}"
        print(msg.format(p == int(y[0]), p, int(y[0])))
