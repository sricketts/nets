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
    x = tf.reshape(x, [-1, n_input])

    # Generate a n_input-element sequence of inputs
    # (eg. [had] [a] [general] -> [20] [6] [33])
    x = tf.split(x,n_input,1)

    # 2-layer LSTM, each layer has n_hidden units.
    # Average Accuracy= 95.20% at 50k iter
    rnn_cell = rnn.MultiRNNCell([rnn.BasicLSTMCell(n_hidden),rnn.BasicLSTMCell(n_hidden)])

    # 1-layer LSTM with n_hidden units but with lower accuracy.
    # Average Accuracy= 90.60% 50k iter
    # Uncomment line below to test but comment out the 2-layer rnn.MultiRNNCell above
    # rnn_cell = rnn.BasicLSTMCell(n_hidden)

    # generate prediction
    outputs, states = rnn.static_rnn(rnn_cell, x, dtype=tf.float32)

    # projection weights and biases
    proj_weights = {
        'out': tf.Variable(tf.random_normal([n_hidden, vocab_size]))
    }
    proj_biases = {
        'out': tf.Variable(tf.random_normal([vocab_size]))
    }
    
    # there are n_input outputs but
    # we only want the last output
    return tf.matmul(outputs[-1], proj_weights['out']) + proj_biases['out']

# tf Graph input
x = tf.placeholder("float", [None, n_input, 1])
y = tf.placeholder("float", [None, vocab_size])

pred = model(x)

print(pred)
