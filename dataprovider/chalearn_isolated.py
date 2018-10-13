import argparse
import os

import cv2
import numpy as np
import tensorflow as tf
from provider import IsolatedSequenceProvider


class ChalearnIsolatedProvider(IsolatedSequenceProvider):
    def __init__(self, root_dir, seq_h, seq_w, seq_l, batch_size):
        """
        Initializes Chalearn dataset with isolated sequences. Due to lack of labelled val and test data for Chalearn, splits it with 9:1 ratio.
        :param root_dir: path to the directory, where 'train' folder and 'train_list.txt' are located
        :param seq_h: the height the images in sequence will be resized to
        :param seq_w: the width the images in sequence will be resized to
        :param seg_l: the length the sequence will be scaled to
        :param batch_size: size of batch
        """
        super().__init__(seq_h, seq_w, seq_l, batch_size)

        # match video names with their class_ids
        description_path = os.path.join(root_dir, 'train_list.txt')
        self._data = self._match_names_labels(description_path, root_dir)

        # split train/val sets
        split_point = int(0.9 * len(self._data))
        self._train_data = self._data[:split_point]
        self._val_data = self._data[split_point:]

    def _read_sequence(self, sequence_path):
        """
        Reads content of the sequence under given sequence_path and converts it to sequence tensor
        :param sequence_path: a path to the directory with sequence
        :return: a tensor of shape [sequence length, image height, image width, image channels]
        """
        sequence = []

        # read avi - from opencv official tutorial
        cap = cv2.VideoCapture(sequence_path)
        while (cap.isOpened()):
            ret, frame = cap.read()
            if ret:
                sequence.append(frame)
            else:
                break
        cap.release()
        cv2.destroyAllWindows()

        sequence_tensor = np.array(sequence, dtype=np.uint8)
        return sequence_tensor

    def _match_names_labels(self, description_file, prefix):
        """
        Reads the file describing Chalearn and matches video paths with class_ids
        :param description_file: path to the description file
        :param prefix: prefix added to filepaths
        :return: a list of pairs (video path, class_id)
        """
        with open(description_file, 'r') as f:
            data = []
            for line in f.readlines():
                elems = line.split(' ')
                path = os.path.join(prefix, elems[0])
                class_id = int(elems[2])
                data.append((path, class_id))
        return data


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate TFRecords related to the Jester dataset')
    parser.add_argument('--root_dir', help='path to directory where "train" folder and "train_list.txt" file are stored', default='../../chalearn_isolated')
    parser.add_argument('--tfrecords_path', help='a path where TFRecords will be stored')
    parser.add_argument('--tfrecords_path', help='a path where TFRecords will be stored')
    args = parser.parse_args()

    provider = ChalearnIsolatedProvider(root_dir=args.root_dir, seq_h=100, seq_w=150, seq_l=60, batch_size=3)
    path = provider._train_data[0][0]
    video = provider._read_sequence(path)

    if args.tfrecords_path is not None:
        provider.generate_tfrecords(args.tfrecords_path, 'Isolated Chalearn')

    # uncomment to check how to fetch data from dataset (and verify wheter it works)
    data_dir = '/media/kpiaskowski/Seagate Backup Plus Drive/Karol_datasets/chalearn_isolated' # path to where tfrecords are stored
    sequence_tensor, class_id, iterator, train_iterator, val_iterator, handle = provider.create_dataset_handles(root_dir=data_dir)

    with tf.Session() as sess:
        # initialize datasets
        train_handle = sess.run(train_iterator.string_handle())
        val_handle = sess.run(val_iterator.string_handle())

        # train
        for i in range(5):
            seq, cls = sess.run([sequence_tensor, class_id], feed_dict={handle: train_handle})
            print('train', i, seq.shape, cls)
        # val
        for i in range(5):
            seq, cls = sess.run([sequence_tensor, class_id], feed_dict={handle: val_handle})
            print('validation', i, seq.shape, cls)
