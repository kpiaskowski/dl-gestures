from collections import deque

import numpy as np
import tensorflow as tf

from constants import jester_num_classes
from dataprovider.jester import JesterProvider
from networks.variable_length import SimpleLSTMNet

# paths
data_dir = "../jester_tfrecords"

sequence_height = 150
sequence_width = 150
sequence_length = 60
batch_size = 5

# dataprovider
provider = JesterProvider(seq_h=sequence_height, seq_w=sequence_width, seq_l=sequence_length, batch_size=batch_size, fake_continuous=True)
num_classes = jester_num_classes
sequence_tensors, labels, iterator, train_iterator, val_iterator, handle = provider.create_dataset_handles(root_dir=data_dir)

# todo tymczasowo dodaje 1 do labelek, zeby nie bylo klasy '-1', bo softmax wariuje - poprawic w innym miejscu (dodac 'blank' klase na koncu slownika)
labels += 1

# define model
is_training = tf.placeholder(tf.bool)
network = SimpleLSTMNet(inputs=sequence_tensors, output_size=num_classes, training_placeholder=is_training, labels=labels)
logits = network.predictions()

onehot = tf.one_hot(labels, num_classes)
loss_weights = tf.to_float(tf.greater_equal(labels, 1))  # todo tutaj tez trzeba bedzie zmienic, gdy blank bedzie na koncu
loss = tf.losses.softmax_cross_entropy(onehot, logits, loss_weights)

train_op = tf.train.RMSPropOptimizer(0.001).minimize(loss)

# session params
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
with tf.Session(config=config) as sess:
    train_handle = sess.run(train_iterator.string_handle())
    val_handle = sess.run(val_iterator.string_handle())
    sess.run(tf.global_variables_initializer())

    last_n_losses = 20
    losses = deque([1000] * last_n_losses, maxlen=last_n_losses)
    for i in range(10000):
        _, cost = sess.run([train_op, loss], feed_dict={handle: train_handle, is_training: True})
        losses.append(cost)
        if i % last_n_losses == 0:
            print(i, np.mean(losses))
