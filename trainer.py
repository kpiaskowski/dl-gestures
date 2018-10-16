import argparse
import inspect
import os
import pathlib
from shutil import copyfile

import tensorflow as tf

from constants import jester_num_classes, chalearn_isolated_num_classes
from dataprovider.jester import JesterProvider
from dataprovider.chalearn_isolated import ChalearnIsolatedProvider

from networks.fixed_length import SimpleConvNet


def dump_training_params(path, args, dataprovider, network):
    """
    Takes a snapshot of training params, provided in CLI args, as well as copies training and dataprovider scripts.
    :param args: CLI arguments - the result of parser.parse_args()
    :param dataprovider: an instance of dataprovider
    :param network_path: an instance of neural network model
    :param path: where snapshot will be stored
    """
    with open(os.path.join(path, "CLI_args.txt"), "w") as text_file:
        print(args, file=text_file)

    dataprovider_path = inspect.getfile(dataprovider.__class__)
    network_path = inspect.getfile(network.__class__)

    copyfile(dataprovider_path, os.path.join(path, dataprovider_path.split('/')[-1]))
    copyfile(network_path, os.path.join(path, network_path.split('/')[-1]))


def run(args):
    # paths
    model_name = args.model_name
    data_dir = args.data_dir
    save_path = os.path.join('../saved_models', model_name)
    pathlib.Path(save_path).mkdir(parents=True, exist_ok=True)  # writers can create nested directories by themselves, saver can't
    train_log_path = os.path.join('../logs', model_name, 'train')
    val_log_path = os.path.join('../logs', model_name, 'val')
    snapshot_path = os.path.join('../saved_models', model_name, 'snapshot')
    pathlib.Path(snapshot_path).mkdir(parents=True, exist_ok=True)

    # choose dataprovider
    if args.dataset_name == 'jester':
        provider = JesterProvider(seq_h=args.sequence_height, seq_w=args.sequence_width, seq_l=args.sequence_length, batch_size=args.batch_size)
        num_classes = jester_num_classes
    elif args.dataset_name == 'chalearn_isolated':
        provider = ChalearnIsolatedProvider(seq_h=args.sequence_height, seq_w=args.sequence_width, seq_l=args.sequence_length, batch_size=args.batch_size)
        num_classes = chalearn_isolated_num_classes
    sequence_tensors, class_ids, iterator, train_iterator, val_iterator, handle = provider.create_dataset_handles(root_dir=data_dir)

    # define model
    is_training = tf.placeholder(tf.bool)
    network = SimpleConvNet(inputs=sequence_tensors, output_size=num_classes, is_training=is_training, labels=class_ids)

    # make a snapshot of the training procedure
    dump_training_params(snapshot_path, args, provider, network)

    # losses and metrics
    loss, accuracy = network.metrics()

    # tensorboard metrics
    tf.summary.scalar('accuracy', accuracy)
    tf.summary.scalar('loss', loss)
    merged = tf.summary.merge_all()

    # training
    eta = args.learning_rate
    validation_ckpt = args.validation_ckpt
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        train_op = tf.train.AdamOptimizer(eta).minimize(loss)

    # session params
    save_ckpt = args.save_ckpt
    saver = tf.train.Saver()
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
        train_writer = tf.summary.FileWriter(train_log_path, sess.graph, flush_secs=60)
        val_writer = tf.summary.FileWriter(val_log_path, flush_secs=60)
        train_handle = sess.run(train_iterator.string_handle())
        val_handle = sess.run(val_iterator.string_handle())
        sess.run(tf.global_variables_initializer())

        i = 0
        while True:
            try:
                if i % save_ckpt == 0 and i > 0:
                    saver.save(sess, global_step=i, save_path=save_path + '/model.ckpt')
                    print('Network saved at step {}'.format(i))

                # training
                for k in range(validation_ckpt):
                    sess.run(train_op, feed_dict={handle: train_handle, is_training: True})
                    i += 1

                # printing stats
                train_loss, train_acc, train_summary = sess.run([loss, accuracy, merged], feed_dict={handle: train_handle, is_training: False})
                val_loss, val_acc, val_summary = sess.run([loss, accuracy, merged], feed_dict={handle: val_handle, is_training: False})
                print('Iteration {}'.format(i))
                print('Training,   loss: {:.6f}, accuracy: {:.2f}%'.format(train_loss, train_acc * 100))
                print('Validation, loss: {:.6f}, accuracy: {:.2f}%'.format(val_loss, val_acc * 100))
                train_writer.add_summary(train_summary, i)
                val_writer.add_summary(val_summary, i)
            except Exception as e:
                print(e)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train the network')
    parser.add_argument('--model_name', help='name of the model and experiment', required=True)
    parser.add_argument('--data_dir', help='path to directory where TFRecords are stored', required=True)
    parser.add_argument('--batch_size', help='number of sequences in single batch', required=True, type=int)

    parser.add_argument('--dataset_name', help='allows to choose different datasets', default='jester', choices=['jester', 'chalearn_isolated'])
    parser.add_argument('--sequence_length', help='all sequences will be adapted to this length, by cropping and stretching', type=int, default=36)
    parser.add_argument('--sequence_height', help='the height the frames in sequences will be resized to', type=int, default=150)
    parser.add_argument('--sequence_width', help='the width the frames in sequences will be resized to', type=int, default=150)

    parser.add_argument('--learning_rate', help='learning_rate', type=float, default=0.0001)
    parser.add_argument('--validation_ckpt', help='how many training steps are being run between validation step', type=int, default=20)
    parser.add_argument('--save_ckpt', help='how many training steps are being run between network saving events', type=int, default=1000)

    args = parser.parse_args()

    run(args)
