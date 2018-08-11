import tensorflow as tf
from tensorflow.contrib import rnn
import numpy as np

import data

dictionary, reverse_dictionary = data.dataset()
vocab_size = len(dictionary)

# Parameters
learning_rate = 0.001
training_iters = 50000
display_step = 1000
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
#y = tf.placeholder("float", [None, vocab_size])

pred = model(x)

init = tf.global_variables_initializer()


with tf.Session() as sess:
    sess.run(init)

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
