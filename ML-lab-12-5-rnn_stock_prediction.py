import tensorflow as tf
import numpy as np
import matplotlib
import os
tf.reset_default_graph()

tf.set_random_seed(777)


import matplotlib.pyplot as plt


def MinMaxScaler(data):
    ''' Min Max Normalization

    Parameters
    ----------
    data : numpy.ndarray
        input data to be normalized
        shape: [Batch size, dimension]

    Returns
    ----------
    data : numpy.ndarry
        normalized data
        shape: [Batch size, dimension]

    References
    ----------
    .. [1] http://sebastianraschka.com/Articles/2014_about_feature_scaling.html

    '''
    numerator = data - np.min(data, 0)
    denominator = np.max(data, 0) - np.min(data, 0)
    # noise term prevents the zero division
    return numerator / (denominator + 1e-7)

seq_length = 7 # 7일치의 주식 정보를 받고, 8일치 주식 시장을 예측하기 위해, 7개를 받는다.
data_dim = 5 # 입력받는 데이터 개수 : 5개
hidden_dim = 10 # 내 마음~
output_dim = 1 # 우리가 원하는 8주차 정보는 한 가지이기 때문
learning_rate = 0.01
iterations = 500

xy = np.loadtxt('data-02-stock_daily.csv', delimiter = ',')
xy = xy[:: -1] # 시간순으로 만들기 위해 ( : : -1 -> 처음부터 끝까지를 거꾸로 가져오기)

train_size = int(len(xy)*0.7)
train_set = xy[0:train_size]
test_set = xy[train_size - seq_length :] # train_size - seq_length ~ 끝까지

# 값들의 차가 너무 크기 때문에 normalization
train_set = MinMaxScaler(train_set)
test_set = MinMaxScaler(test_set)

def build_dataset(time_series, seq_length):
    dataX = []
    dataY = []
    for i in range(0, len(time_series) - seq_length):
        _x = time_series[i:i + seq_length, :]
        _y = time_series[i + seq_length, [-1]]
        print(_x, '->', _y)
        dataX.append(_x)
        dataY.append(_y)
    return np.array(dataX), np.array(dataY)

trainX, trainY = build_dataset(train_set, seq_length)
testX, testY = build_dataset(test_set, seq_length)

X = tf.placeholder(tf.float32, [None, seq_length, data_dim])
Y = tf.placeholder(tf.float32, [None, 1]) # 8주차의 결과 즉 1개의 값 만 원하기 때문

cell = tf.contrib.rnn.BasicLSTMCell(num_units = hidden_dim, state_is_tuple = True, activation = tf.tanh)
outputs, _states = tf.nn.dynamic_rnn(cell, X, dtype = tf.float32)
# y 값들을 모아서 fc 진행
Y_pred = tf.contrib.layers.fully_connected( outputs[:,-1], output_dim, activation_fn = None)

loss = tf.reduce_sum(tf.square(Y_pred - Y))

optimizer = tf.train.AdamOptimizer(learning_rate)
train = optimizer.minimize(loss)

targets = tf.placeholder(tf.float32, [None, 1])
predictions = tf.placeholder(tf.float32, [None, 1])
rmse = tf.sqrt(tf.reduce_mean(tf.square(targets - predictions))) # 제곱의 평균을 구한 뒤 그 제곱근을 계산

with tf.Session() as sess:
    init = tf.global_variables_initializer()
    sess.run(init)
    
    for i in range(iterations):
        _, step_loss = sess.run([train, loss], feed_dict ={ X : trainX, Y : trainY})
        print( "[step : {}] loss : {}".format(i, step_loss))
        
        
    test_predict = sess.run(Y_pred, feed_dict = {X: testX})
    rmse_val = sess.run(rmse, feed_dict = { targets : testY, predictions : test_predict})
    print("RMSE : {}".format(rmse_val))
    
    plt.plot(testY)
    plt.plot(test_predict)
    plt.xlabel("Time Period")
    plt.ylabel("Stock Price")
    plt.show()