from flask import Flask, jsonify, request

import tensorflow as tf

app = Flask(__name__ )

@app.route('/')
def home_page():
   return 'Train a deep learning model'

# Input
W = tf.Variable([.26], dtype=tf.float32)
b = tf.Variable([-.26], dtype=tf.float32)

# Output
curr_W, curr_b, curr_loss = 0.0, 0.0, 0.0

# Initializes model parameters
@app.route('/initialize-model')
def initialize():
    global W, b
    W = tf.Variable([.26], dtype=tf.float32)
    b = tf.Variable([-.26], dtype=tf.float32)
    return 'Initialized model parameters - W: %s b: %s' % (W, b)


@app.route('/train-model')
def train_model():
    global W, b, curr_W, curr_b, curr_loss
    # Input and Output
    x = tf.placeholder(tf.float32)
    linear_model = W * x + b
    y = tf.placeholder(tf.float32)

    # loss
    loss = tf.reduce_sum(tf.square(linear_model - y))  # sum of the squares
    # optimizer
    optimizer = tf.train.GradientDescentOptimizer(0.01)
    train = optimizer.minimize(loss)

    # training data
    x_train = [1, 2, 3, 4]
    y_train = [0, -1, -2, -3]
    # training loop
    init = tf.global_variables_initializer()
    sess = tf.Session()
    sess.run(init)  # Reset values
    for i in range(1000):
        sess.run(train, {x: x_train, y: y_train})

    # Evaluate training accuracy
    curr_W, curr_b, curr_loss = sess.run([W, b, loss], {x: x_train, y: y_train})
    return 'Finished training and evaluation'


@app.route('/output-result')
def output_result():
    global curr_W, curr_b, curr_loss
    return "Result of training the data - W: %s b: %s loss: %s" % (curr_W, curr_b, curr_loss)

# return jsonify({'Result of evaluation': language_list})

if __name__ == '__main__':
   app.run(host = '127.0.0.1', port = 5000, debug = True)