import os

import tensorflow as tf

from constants import jester_num_classes
from dataprovider.jester import JesterProvider
from networks.variable_length import SimpleLSTMNet

# paths
data_dir = "../jester_tfrecords"

sequence_height = 150
sequence_width = 150
sequence_length = 60
batch_size = 10

# dataprovider
provider = JesterProvider(seq_h=sequence_height, seq_w=sequence_width, seq_l=sequence_length, batch_size=batch_size, fake_continuous=True)
num_classes = jester_num_classes
sequence_tensors, class_ids, iterator, train_iterator, val_iterator, handle = provider.create_dataset_handles(root_dir=data_dir)

# define model
is_training = tf.placeholder(tf.bool)
network = SimpleLSTMNet(inputs=sequence_tensors, output_size=num_classes, training_placeholder=is_training, labels=class_ids)
net_out = network.predictions()

# session params
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
with tf.Session(config=config) as sess:
    train_handle = sess.run(train_iterator.string_handle())
    val_handle = sess.run(val_iterator.string_handle())
    sess.run(tf.global_variables_initializer())

    for i in range(1):
        seq, cls = sess.run([net_out, class_ids], feed_dict={handle: train_handle, is_training: True})
        print(seq.shape)
