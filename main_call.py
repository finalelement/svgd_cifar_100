import os
import pickle
import random
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from optimizer import SVGD
from utils import Time

def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

def network(inputs, labels, scope):
    num_class = 100
    def init_weights(shape):
        init_random_dist = tf.truncated_normal(shape, stddev=0.1)
        return tf.Variable(init_random_dist)

    def init_bias(shape):
        init_bias_vals = tf.constant(0.1, shape=shape)
        return tf.Variable(init_bias_vals)

    def conv2d(x, W):
        return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

    def max_pool_2by2(x):
        return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                              strides=[1, 2, 2, 1], padding='SAME')

    def convolutional_layer(input_x, shape):
        W = init_weights(shape)
        b = init_bias([shape[3]])
        return tf.nn.relu(conv2d(input_x, W) + b)

    def normal_full_layer(input_layer, size):
        input_size = int(input_layer.get_shape()[1])
        W = init_weights([input_size, size])
        b = init_bias([size])
        return tf.matmul(input_layer, W) + b


    #hold_prob = tf.placeholder(tf.float32)

    # See /derivations/bayesian_classification.pdf for mathematical details.
    with tf.variable_scope(scope):
        #for _ in range(2):
        #    net = tf.layers.dense(net, 100, activation=tf.nn.tanh)
        convo_1 = convolutional_layer(inputs, shape=[3, 3, 3, 32])

        convo_2 = convolutional_layer(convo_1, shape=[3, 3, 32, 64])
        convo_2_pooling = max_pool_2by2(convo_2)

        convo_3 = convolutional_layer(convo_2_pooling, shape=[3, 3, 64, 128])
        convo_4 = convolutional_layer(convo_3, shape=[3, 3, 128, 256])
        convo_4_pooling = max_pool_2by2(convo_4)

        convo_2_flat = tf.reshape(convo_4_pooling, [-1, 8 * 8 * 256])
        full_layer_one = tf.nn.relu(normal_full_layer(convo_2_flat, 1024))
        #full_one_dropout = tf.nn.dropout(full_layer_one, hold_prob)
        logits = normal_full_layer(full_layer_one, num_class)

        #logits = tf.layers.dense(net, 1)
        log_likelihood = - tf.nn.softmax_cross_entropy_with_logits(labels=labels, logits=logits)
        variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=scope)
        prob_1_x_w = tf.nn.softmax(logits)
        gradients = tf.gradients(log_likelihood, variables)
    return gradients, variables, prob_1_x_w

def make_gradient_optimizer():
    return tf.train.AdamOptimizer(learning_rate=0.001)

def main():

    num_class = 100
    num_particles = 5
    num_iterations = 10
    batch_size = 1000
    algorithm = 'svgd'

    data_path = r'D:\Users\Vishwesh\Datasets\cifar-100-python\cifar-100-python\train'
    data_path = os.path.normpath(data_path)

    meta_path = r'D:\Users\Vishwesh\Datasets\cifar-100-python\cifar-100-python\meta'
    meta_path = os.path.normpath(meta_path)

    # Load Meta data using pickle
    meta_dict = unpickle(meta_path)
    print(meta_dict.keys())

    # Load Data using pickle
    data_dict = unpickle(data_path)
    print(data_dict.keys())

    # Grab the Data and the fine labels
    train_data = data_dict[b'data']
    train_fine_labels = data_dict[b'fine_labels']
    all_labels = np.eye(100)[train_fine_labels]

    # Reshape the training data
    images = list()
    for d in train_data:
        image = np.zeros((32,32,3), dtype=np.uint8)
        image[...,0] = np.reshape(d[:1024], (32, 32)) # Red Channel
        image[...,1] = np.reshape(d[1024:2048], (32, 32))  # Green Channel
        image[...,2] = np.reshape(d[2048:], (32, 32))  # Blue Channel
        images.append(image)

    images = np.asarray(images, dtype=float)
    images = images/255.0

    x = tf.placeholder(tf.float32, shape=[None, 32, 32, 3])
    y_true = tf.placeholder(tf.float32, shape=[None, num_class])
    grads_list, vars_list, prob_1_x_w_list = [], [], []

    for i in range(num_particles):
        grads, vars, prob_1_x_w = network(x, y_true, 'p{}'.format(i))
        grads_list.append(grads)
        vars_list.append(vars)
        prob_1_x_w_list.append(prob_1_x_w)

    if algorithm == 'svgd':
        optimizer = SVGD(grads_list=grads_list,
                         vars_list=vars_list,
                         make_gradient_optimizer=make_gradient_optimizer)

    prob_1_x = tf.reduce_mean(tf.stack(prob_1_x_w_list), axis=0)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        with Time("training"):
            for _ in range(num_iterations):
                for bs in range(0,len(images),batch_size):
                    start = bs
                    end_c = bs+batch_size
                    x_train = images[start:end_c]
                    y_train = all_labels[start:end_c]
                    cross_entropy = -tf.reduce_sum(y_true * tf.log(prob_1_x))
                    _, loss_val = sess.run([optimizer.update_op, cross_entropy], feed_dict={x: x_train, y_true: y_train})
                    print ('loss = ' + str(loss_val))
    print('Debug here')
    return None

if __name__ == "__main__":
    main()