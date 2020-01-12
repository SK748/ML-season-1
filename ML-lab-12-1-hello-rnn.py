# input hihello -> output ihello
import tensorflow as tf
import numpy as np
tf.set_random_seed(777)
tf.reset_default_graph()

idx2char = ['h', 'i', 'e', 'l', 'o']

x_data = [[0, 1, 0, 2, 3, 3]] # hihell, 입력 값이기 때문에 마지막인 o는 필요 없음
x_one_hot = [[[1,0,0,0,0],[0,1,0,0,0],[1,0,0,0,0],[0,0,1,0,0],[0,0,0,1,0],[0,0,0,1,0]]] #hihell을 one_hot 한 것

y_data = [[1, 0, 2, 3, 3, 4]] # ihello로 출력

# idx2char의 수, 즉 unique char의 수
num_classes = 5
# input_dim : 입력 시 dim의 크기, 즉 x_one_hot size
input_dim = 5 
# hidden_size : output size, 보통 사용자 지정이지만, 이 문제에서는 one_hot으로 출력받고 싶어해 5
hidden_size = 5
# batch_size : 사용 할 문장의 수(입력할 데이터의 개수)
batch_size = 1
# sequence_length : 입력받을 문자의 길이 여기선 hihell = 6
sequence_length = 6
learning_rate = 0.1

# X one_hot, shape = [batch_size, seq_length, input_dim] 인데 batch_size의 경우 몇 개 들어와도 상관 없어서 None으로 표시
X = tf.placeholder(tf.float32, [None, sequence_length, input_dim])
# Y label
Y = tf.placeholder(tf.int32, [None, sequence_length])

# 셀을 생성시키는 부분, hidden_size에 맞춰 output의 shape가 결정된다.
cell = tf.contrib.rnn.BasicLSTMCell(num_units = hidden_size, state_is_tuple = True)
initial_state = cell.zero_state(batch_size, tf.float32)
# 셀을 학습시키는 부분, cell과 X data를 넣어주면, 예상 값(outputs)를 만들어 주는 함수
# outputs shape = (1, 6, 5)
outputs, _states = tf.nn.dynamic_rnn(cell, X, initial_state = initial_state, dtype = tf.float32)

# FC layer
X_for_fc = tf.reshape(outputs, [-1, hidden_size])
# fc_w = tf.get_variable("fc_w", [hidden_size, num_classes])
# fc_b = tf.get_variable("fc_b", [num_classes])
# outputs = tf.matmul(X_for_fo, fc_w) + fc_b
outputs = tf.contrib.layers.fully_connected(inputs = X_for_fc, num_outputs = num_classes, activation_fn = None)

# reshape out for sequence_loss
outputs = tf.reshape(outputs, [batch_size, sequence_length, num_classes])

# sequence_loss에서 각 중요도를 나타냄, 차이가 없으면 모두 1로 표현
weights = tf.ones([batch_size, sequence_length])
# 예상값 outputs과 정답 Y를 받아 두 값의 loss를 구하는 함수
sequence_loss = tf.contrib.seq2seq.sequence_loss(logits = outputs, targets = Y, weights = weights)
loss = tf.reduce_mean(sequence_loss)
train = tf.train.AdamOptimizer(learning_rate = learning_rate).minimize(loss)

# one_hot 안의 값을 argmax해야함 따라서 axis = 2
prediction = tf.argmax(outputs, axis = 2)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(50):
        I, _ = sess.run([loss, train], feed_dict = { X: x_one_hot, Y: y_data})
        result = sess.run(prediction, feed_dict = {X: x_one_hot})
        print(i, "loss :", I, "prediction :", result, "true Y: ", y_data)
        
        # print char using dic
        result_str = [idx2char[c] for c in np.squeeze(result)]
        print("\tPrediction str: ", ''.join(result_str))