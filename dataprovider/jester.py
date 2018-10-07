import os
import tensorflow as tf
import cv2
import numpy as np

from dataprovider.provider import IsolatedSequenceProvider, TFRecordWriter


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

    def read_sequence(self, dir_name):
        """
        Reads content of the sequence under given dir name and converts it to tensor
        :param dir_name: a path to the directory with sequence images
        :return: a tensor of shape [sequence length, image height, image width, image channels]
        """
        image_names = sorted([os.path.join(dir_name, name) for name in os.listdir(dir_name)])
        images = [cv2.imread(name) for name in image_names]
        sequence_tensor = np.array(images, dtype=np.uint8)
        return sequence_tensor

    def generate_tfrecords(self, root_dir):
        """
        Generates data in the form of TF records
        :param root_dir: root directory, where 'train' and 'validation' data will be stored.
        """
        # 100 sequences per single TFRecord
        writer = TFRecordWriter(root_dir, record_length=100, seq_reading_func=self.read_sequence, is_isolated=True)

        writer.generate_tfrecords(self._train_data, 'train', 'Jester')
        writer.generate_tfrecords(self._val_data, 'val', 'Jester')


if __name__ == '__main__':
    provider = JesterProvider(data_dir='../../jester/data',
                              csv_dir='../../jester/csv',
                              seq_h=100, seq_w=150, seq_l=60,
                              batch_size=1)
    sequence_tensor, class_id, iterator = provider.create_dataset_handles('jester_data')

    with tf.Session() as sess:
        sess.run(iterator.initializer)

        seq, cls = sess.run([sequence_tensor, class_id])
        print(seq.shape)
        for s in seq:
            for i, img in enumerate(s):
                cv2.imshow('', img)
                cv2.waitKey(100)
