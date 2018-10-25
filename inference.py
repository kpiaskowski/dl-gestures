import argparse
from collections import deque

import cv2
import inspect
import os
import pathlib
import sys
from shutil import copyfile

import numpy as np
import tensorflow as tf
from tensorflow.python.ops.image_ops_impl import ResizeMethod

from dataprovider.chalearn_isolated import ChalearnIsolatedProvider
from dataprovider.jester import JesterProvider
from networks.fixed_length import SimpleConvNet


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


def standardize(sequence_tensor):
    """
    Scales image to <0...1>, then standardizes it. Standardization is based on tf.image.per_image_standardization (but allows tensors of arbitrary shape),
    so it should be numerically safe.
    :return: standardized sequence
    """
    # scale sequence_tensor to 0 ... 1
    sequence_tensor = sequence_tensor / 255

    # compute statistics
    mean = tf.reduce_mean(sequence_tensor)
    variance = tf.reduce_mean(tf.square(sequence_tensor)) - tf.square(tf.reduce_mean(sequence_tensor))
    variance = tf.nn.relu(variance)  # avoid nagative values for numerical safety
    stddev = tf.sqrt(variance)

    # Apply a minimum normalization that protects us against uniform images (taken from tf.image.per_image_standardization)
    num_pixels = tf.reduce_prod(tf.shape(sequence_tensor))  # number of pixels in image
    min_stddev = tf.math.rsqrt(tf.cast(num_pixels, tf.float32))  # 1/sqrt, 'fast reciprocal square root'
    pixel_value_scale = tf.maximum(stddev, min_stddev)
    pixel_value_offset = mean
    standardized_sequence = tf.subtract(sequence_tensor, pixel_value_offset)
    standardized_sequence = tf.div(standardized_sequence, pixel_value_scale)

    return standardized_sequence


def resize_spatially(sequence_tensor, seq_h, seq_w):
    """
    Resizes sequence tensor in spatial dimensions (h, w), using bilinear interpolation
    :return: sequence resized spatially
    """
    resized_sequence = tf.image.resize_images(sequence_tensor, size=(seq_h, seq_w))
    return resized_sequence


def stretch_temporally(sequence_tensor, new_length, seq_w):
    """
    Stretches sequence temporally to the desired length, using NNI.
    :return: sequence resized temporally
    """
    # swap [l, h, w, c] to  [h, l, w, c]
    sequence = tf.transpose(sequence_tensor, perm=[1, 0, 2, 3])
    sequence = tf.image.resize_images(sequence, size=(new_length, seq_w), method=ResizeMethod.NEAREST_NEIGHBOR)

    # swap back to [l, h, w, c]
    sequence = tf.transpose(sequence, perm=[1, 0, 2, 3])
    return sequence


def run(model_path, num_classes, class_names, seq_l, seq_h, seq_w, channels):
    """
    Loads neural network (depending on the dataset used, because num_classes vary across datasets) and runs the live video feed and inference
    :param model_path: path to the directory, where checkpoints are located. The system will try to find the most recent one.
    :param num_classes: number of classes in the dataset
    :param class_names: list with class names
    :param seq_l: lengths of sequences
    :param seq_h: heights of sequences
    :param seq_w: widths of sequences
    :param channels: number of channels in sequences
    """
    # input placeholders
    sequence_placeholder = tf.placeholder(tf.float32, [None, None, None, None])
    is_training = tf.placeholder(tf.bool)

    # preprocessing pipeline
    data = standardize(sequence_placeholder)
    data = resize_spatially(data, seq_h, seq_w)
    data = tf.reshape(data, [seq_l, seq_h, seq_w, channels])  # tf hack to pass dimensionality
    data = tf.expand_dims(data, 0)  # add artificial first dimension (since only one example is being processed at a time)

    network = SimpleConvNet(inputs=data, output_size=num_classes, is_training=is_training, labels=None)
    predictions = network.predictions()

    saver = tf.train.Saver()
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        try_restore(model_path, sess, saver)

        # live video feed
        buffer = deque(maxlen=seq_l)  # fifo queue of seq_l length
        cap = cv2.VideoCapture(0)
        ret = True
        previous_gesture = None
        counter = 0
        while ret:
            ret, frame = cap.read()
            buffer.append(frame)
            if len(buffer) < seq_l:  # prefill buffer to have seq_l elements
                continue

            cv2.imshow('inference', buffer[-1])
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            if counter % 4 == 0:  # predict only every 4th frame
                outputs = sess.run(predictions, feed_dict={sequence_placeholder: buffer, is_training: False})
                gesture = class_names[np.argmax(outputs[0])]
                if gesture != previous_gesture:
                    print(gesture)
                    previous_gesture = gesture
            counter += 1

        # finally release all
        cap.release()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    dataset = 'jester'
    # dataset = 'chalearn_isolated'

    if dataset == 'jester':
        model_path = '../best_saved_models/jester_simple_conv'
        num_classes = 27
    elif dataset == 'chalearn_isolated':
        model_path = '../best_saved_models/chalearn_isolated_simple_conv'
        num_classes = 250
    seq_l, seq_h, seq_w, channels = 36, 150, 150, 3  # values used during training
    class_names = JesterProvider(None, None, None, None).classes  # create dataset only to get classes

    run(model_path, num_classes, class_names, seq_l, seq_h, seq_w, channels)
