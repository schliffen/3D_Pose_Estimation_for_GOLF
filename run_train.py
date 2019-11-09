#
#
import numpy as np
import tensorflow as tf

import argparse

args = argparse.ArgumentParser()
args.add_argument('-d0', '--', default='', help='' )
args.add_argument('-d1', '--', default='', help='' )
args.add_argument('-d2', '--', default='', help='' )
args.add_argument('-d3', '--', default='', help='' )
args.add_argument('-d4', '--', default='', help='' )
args.add_argument('-d5', '--', default='', help='' )


# tf Graph input
X = tf.placeholder("float", [None, n_input])
Y = tf.placeholder("float", [None, n_classes])
mlp_layer_name = ['h1', 'b1', 'h2', 'b2', 'h3', 'b3', 'w_o', 'b_o']
logits = multilayer_perceptron(X, n_input, n_classes, mlp_layer_name)
loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=Y), name='loss_op')
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
train_op = optimizer.minimize(loss_op, name='train_op')

with tf.Session() as sess:
    saver = tf.train.Saver()
    saver.restore(sess, tf.train.latest_checkpoint('./')) # search for checkpoint file

    graph = tf.get_default_graph()

    for epoch in range(training_epochs):
        avg_cost = 0.

        # Loop over all batches
        for i in range(total_batch):
            batch_x, batch_y = next(train_generator)

            # Run optimization op (backprop) and cost op (to get loss value)
            _, c = sess.run([train_op, loss_op], feed_dict={X: batch_x,
                                                            Y: batch_y})
            # Compute average loss
            avg_cost += c / total_batch

        print("Epoch: {:3d}, cost = {:.6f}".format(epoch+1, avg_cost))
