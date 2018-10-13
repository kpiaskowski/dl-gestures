import os
import pathlib
import random

import tensorflow as tf

from constants import jester_num_classes
from dataprovider.jester import JesterProvider
from networks.fixed_length import SimpleConvNet

# paths
model_name = 'jester_test' + str(random.randint(0, 10000000))
data_dir = '/media/kpiaskowski/Seagate Backup Plus Drive/Karol_datasets/jester_data'
train_log_path = os.path.join('logs', model_name, 'train')
val_log_path = os.path.join('logs', model_name, 'val')
save_path = os.path.join('saved_models', model_name, 'model.ckpt')
pathlib.Path(save_path).mkdir(parents=True, exist_ok=True)  # writers can create nested directories by themselves, saver can't

# define dataprovider
provider = JesterProvider(seq_h=150, seq_w=150, seq_l=36, batch_size=7)
sequence_tensors, class_ids, iterator, train_iterator, val_iterator, handle = provider.create_dataset_handles(root_dir=data_dir)

# define model
is_training = tf.placeholder(tf.bool)
network = SimpleConvNet(inputs=sequence_tensors, output_size=jester_num_classes, is_training=is_training, labels=class_ids)

# losses and metrics
prediction_loss, accuracy = network.metrics()
regulatization_loss = tf.reduce_sum(tf.losses.get_regularization_losses())
loss = prediction_loss + regulatization_loss

# tensorboard metrics
tf.summary.scalar('accuracy', accuracy)
tf.summary.scalar('loss', loss)
merged = tf.summary.merge_all()

# training
eta = 0.0001
validation_ckpt = 20
save_ckpt = 50
update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
with tf.control_dependencies(update_ops):
    train_op = tf.train.AdamOptimizer(eta).minimize(loss)

# session params
saver = tf.train.Saver(keep_checkpoint_every_n_hours=2)
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
with tf.Session(config=config) as sess:
    train_writer = tf.summary.FileWriter(train_log_path, sess.graph, flush_secs=30)
    val_writer = tf.summary.FileWriter(val_log_path, flush_secs=30)
    train_handle = sess.run(train_iterator.string_handle())
    val_handle = sess.run(val_iterator.string_handle())
    sess.run(tf.global_variables_initializer())

    i = 0
    while True:
        if i % save_ckpt == 0 and i > 0:
            saver.save(sess, global_step=i, save_path=save_path)
            pass

        # training
        for k in range(validation_ckpt):
            sess.run(train_op, feed_dict={handle: train_handle, is_training: True})
            i += 1

        # printing stats
        train_loss, train_acc = sess.run([loss, accuracy], feed_dict={handle: train_handle, is_training: False})
        val_loss, val_acc = sess.run([loss, accuracy], feed_dict={handle: val_handle, is_training: False})
        print('Iteration {}'.format(i))
        print('Training,   loss: {:.6f}, accuracy: {:.2f}%'.format(train_loss, train_acc * 100))
        print('Validation, loss: {:.6f}, accuracy: {:.2f}%'.format(val_loss, val_acc * 100))

# todo obuduj WSZYZSTKO w arg.param
# todo przenies moze trainer do jakiegos osobnego pliku/brancha?