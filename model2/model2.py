import tensorflow as tf
from tensorflow.contrib import rnn
import numpy as np

import data

# Text file containing words for training
training_file = 'belling_the_cat.txt'

training_data = data.read_data(training_file)
dictionary, reverse_dictionary = data.build_dataset(training_data)
vocab_size = len(dictionary)

# Parameters
learning_rate = 0.001
training_iters = 50000
#training_iters = 5000
display_iters = 1000
n_input = 3

# number of units in RNN cell
n_hidden = 512

def model(x):

    # reshape to [1, n_input]
    #x = tf.reshape(x, [-1, n_input])

    # Generate a n_input-element sequence of inputs
    # (eg. [had] [a] [general] -> [20] [6] [33])
    x2 = tf.split(x,n_input,1)

    # 2-layer LSTM, each layer has n_hidden units.
    rnn_cell = rnn.MultiRNNCell([rnn.BasicLSTMCell(n_hidden),
                                 rnn.BasicLSTMCell(n_hidden)])

    # generate prediction
    outputs, states = rnn.static_rnn(rnn_cell, x2, dtype=tf.float32)

    # projection weights and biases
    proj_weights = tf.Variable(tf.random_normal([n_hidden, vocab_size]))
    proj_biases = tf.Variable(tf.random_normal([vocab_size]))
    
    # there are n_input outputs but
    # we only want the last output
    return tf.matmul(outputs[-1], proj_weights) + proj_biases

# tf Graph input
x = tf.placeholder("float", [1, n_input])
y = tf.placeholder("float", [1, vocab_size])

pred = model(x)

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
optimizer = tf.train.RMSPropOptimizer(learning_rate=learning_rate).minimize(cost)

# Model evaluation
correct_pred = tf.equal(tf.argmax(pred,1), tf.argmax(y,1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

init = tf.global_variables_initializer()

total_loss = 0
total_acc = 0

with tf.Session() as sess:
    sess.run(init)


    # Training

    print("Training")
    training_datapoints = len(training_data) // n_input
    if len(training_data) % n_input == 0:
        training_datapoints = training_datapoints - 1
    for step in range(training_iters):
        offset = (step % training_datapoints) * n_input
        symbols_in_keys = [dictionary[str(training_data[i])] for i in range(offset, offset+n_input) ]
        labels_onehot = np.zeros([vocab_size], dtype=float)
        labels_onehot[dictionary[str(training_data[offset+n_input])]] = 1.0
        _, acc, loss, onehot_pred = sess.run([optimizer, accuracy, cost, pred],
                                                feed_dict={x: [symbols_in_keys], y: [labels_onehot]})
        total_acc += acc
        total_loss += loss
        if (step+1) % display_iters == 0:
            print("After %d steps" % (step+1))
            print("\tavg acc = %f" % (total_acc/display_iters))
            print("\tavg loss = %f" % (total_loss/display_iters))
            total_acc = 0
            total_loss = 0
        
    # Inference Demo
    
    input_sentence = "if i will"
    sentence = input_sentence
    input_words = input_sentence.split()
    output_len = 32
    symbols_in_keys = [dictionary[str(input_words[i])] for i in range(len(input_words))]
    for i in range(output_len):
        onehot_pred = sess.run(pred, feed_dict={x: [symbols_in_keys]})
        onehot_pred_index = int(tf.argmax(onehot_pred, 1).eval())
        sentence = "%s %s" % (sentence,reverse_dictionary[onehot_pred_index])
        symbols_in_keys = symbols_in_keys[1:]
        symbols_in_keys.append(onehot_pred_index)
    print(sentence)
