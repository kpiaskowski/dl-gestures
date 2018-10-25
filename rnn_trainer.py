import argparse
import inspect
import os
import pathlib
import sys
from shutil import copyfile

import tensorflow as tf

from dataprovider.chalearn_isolated import ChalearnIsolatedProvider
from dataprovider.jester import JesterProvider
from networks.variable_length import SimpleLSTMNet


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
    trainer_path = __file__

    copyfile(dataprovider_path, os.path.join(path, dataprovider_path.split('/')[-1]))
    copyfile(network_path, os.path.join(path, network_path.split('/')[-1]))
    copyfile(trainer_path, os.path.join(path, trainer_path.split('/')[-1]))


def try_restore(save_path, sess, saver):
    """
    Checks whethere there is an existing checkpoint with saved weights and tries to restore it. Also computes the latest saved global step, which could be used later as an
    offset to the current step counter
    :param save_path: path where checkpoints should be located
    :param sess: tensorflow session
    :param saver: instance of saver (for restoring)
    :return: offset as number (0 if no checkpoint was found)
    """
    ckpt = tf.train.latest_checkpoint(save_path)
    if ckpt is not None:
        # find latest saved global_step, which will be used as an offset to the current step counter
        offset = int(ckpt.split('-')[-1])
        saver.restore(sess, ckpt)
        # try to load checkpoint
        print('Loaded model from {}'.format(ckpt))
    else:
        offset = 0
    return offset


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
        provider = JesterProvider(seq_h=args.sequence_height, seq_w=args.sequence_width, seq_l=args.sequence_length, batch_size=args.batch_size, fake_continuous=True)
    elif args.dataset_name == 'chalearn_isolated':
        provider = ChalearnIsolatedProvider(seq_h=args.sequence_height, seq_w=args.sequence_width, seq_l=args.sequence_length, batch_size=args.batch_size, fake_continuous=True)
    sequence_tensors, labels, iterator, train_iterator, val_iterator, handle = provider.create_dataset_handles(root_dir=data_dir)

    # define model
    is_training = tf.placeholder(tf.bool)
    network = SimpleLSTMNet(inputs=sequence_tensors, num_classes=provider.num_classes(), training_placeholder=is_training, labels=labels)

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
        optimizer = tf.train.AdamOptimizer(eta)
        gradients, variables = zip(*optimizer.compute_gradients(loss))
        gradients = [None if gradient is None else tf.clip_by_norm(gradient, 3.0) for gradient in gradients]
        train_op = optimizer.apply_gradients(zip(gradients, variables))

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

        # instead of relying on chance to hit exact step during saving, let's always check whether step // save_ckpt yields the same number as in previous step. If not,
        # save model and update the result of the last_saved_idx
        step = try_restore(save_path, sess, saver)
        last_saved_idx = step // save_ckpt

        while True:
            try:
                if step // save_ckpt != last_saved_idx and step > 0:
                    last_saved_idx = step // save_ckpt
                    saver.save(sess, global_step=step, save_path=save_path + '/model.ckpt')
                    print('Network saved at step {}'.format(step))

                # training
                for k in range(validation_ckpt):
                    sess.run(train_op, feed_dict={handle: train_handle, is_training: True})
                    step += 1

                # printing stats
                train_loss, train_acc, train_summary = sess.run([loss, accuracy, merged], feed_dict={handle: train_handle, is_training: False})
                val_loss, val_acc, val_summary = sess.run([loss, accuracy, merged], feed_dict={handle: val_handle, is_training: False})
                print('Iteration {}'.format(step))
                print('Training,   loss: {:.6f}, accuracy: {:.2f}%'.format(train_loss, train_acc * 100))
                print('Validation, loss: {:.6f}, accuracy: {:.2f}%'.format(val_loss, val_acc * 100))
                train_writer.add_summary(train_summary, step)
                val_writer.add_summary(val_summary, step)
            except KeyboardInterrupt:
                # attempt to save model on exit
                print('Keyboard interrupt, trying to save model...')
                saver.save(sess, global_step=step, save_path=save_path + '/model.ckpt')
                print('Model saved!')
                sys.exit(0)
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
