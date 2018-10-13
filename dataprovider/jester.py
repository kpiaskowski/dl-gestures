import argparse
import os
import tensorflow as tf
import cv2
import numpy as np
from provider import IsolatedSequenceProvider


class JesterProvider(IsolatedSequenceProvider):
    def __init__(self, data_dir, csv_dir, seq_h, seq_w, seq_l, batch_size):
        """
        Initializes Jester dataset.
        :param data_dir: path to root directory with data
        :param csv_dir: path to root directory with csv_files
        :param seq_h: the height the images in sequence will be resized to
        :param seq_w: the width the images in sequence will be resized to
        :param seg_l: the length the sequence will be scaled to
        :param batch_size: size of batch
        """
        super().__init__(seq_h, seq_w, seq_l, batch_size)
        self._data_dir = data_dir

        # read classes available in the Jester dataset
        self._classes = self._read_classes(os.path.join(csv_dir, 'jester-v1-labels.csv'))

        # generate train and validation pairs of (image name, label)
        self._train_data = self._match_names_labels(csv_path=os.path.join(csv_dir, 'jester-v1-train.csv'))
        self._val_data = self._match_names_labels(csv_path=os.path.join(csv_dir, 'jester-v1-validation.csv'))

    def _read_classes(self, csv_path):
        """
        Creates a list of classes available in Jester dataset.
        :param csv_path: path to the csv file describing classes
        :return: a list of Jester classes
        """
        with open(csv_path, 'r') as f:
            classes = sorted([name.rstrip('\n') for name in f.readlines()])
            return classes

    def _match_names_labels(self, csv_path):
        """
        For given CSV file (from training/validation set), makes a list of directories with sequences and their corresponding labels.
        :param csv_path: path to the respective csv file
        :return: a list of tuples: (directory path, corresponding class label)
        """

        # read CSV file
        def parse_csv_line(line):
            """Parses a single line from CSV file from Jester dataset"""
            sequence_id, cls = line.split(';')
            sequence_path = os.path.join(self._data_dir, sequence_id)
            class_id = self._classes.index(cls.rstrip('\n'))
            return sequence_path, class_id

        with open(csv_path, 'r') as f:
            lines = [parse_csv_line(line) for line in f.readlines()]  # (sequence path, class id) for each line
            return lines

    def _read_sequence(self, dir_name):
        """
        Reads content of the sequence under given dir name and converts it to tensor
        :param dir_name: a path to the directory with sequence images
        :return: a tensor of shape [sequence length, image height, image width, image channels]
        """
        image_names = sorted([os.path.join(dir_name, name) for name in os.listdir(dir_name)])
        images = [cv2.imread(name) for name in image_names]
        sequence_tensor = np.array(images, dtype=np.uint8)
        return sequence_tensor


# only for looking how to run dataset and get data
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate TFRecords related to the Jester dataset')
    parser.add_argument('--data_dir', help='path to the data where parent folder, where Jester folders with png images are stored', default='../../jester/data')
    parser.add_argument('--csv_dir', help='path to the data where parent folder, where Jester csv files are stored', default='../../jester/csv')
    parser.add_argument('--tfrecords_path', help='a path where TFRecords will be stored')
    args = parser.parse_args()

    provider = JesterProvider(data_dir=args.data_dir,
                              csv_dir=args.csv_dir,
                              seq_h=100, seq_w=150, seq_l=60,
                              batch_size=3)

    if args.tfrecords_path is not None:
        provider.generate_tfrecords(args.tfrecords_path, 'Jester')

    # uncomment to check how to fetch data from dataset (and verify wheter it works)
    data_dir = '/media/kpiaskowski/Seagate Backup Plus Drive/Karol_datasets/jester_data' # path to where tfrecords are stored
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
