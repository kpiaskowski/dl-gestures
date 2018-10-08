import os
import tensorflow as tf
import cv2
import numpy as np
import argparse
from provider import IsolatedSequenceProvider, TFRecordWriter


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
                path = os.path.join(prefix, elems[1])
                class_id = int(elems[2])
                data.append((path, class_id))
        return data

    def generate_tfrecords(self, root_dir):
        """
        Generates data in the form of TF records
        :param root_dir: root directory, where 'train' and 'validation' data will be stored.
        """
        # 100 sequences per single TFRecord
        writer = TFRecordWriter(root_dir, record_length=100, seq_reading_func=self._read_sequence, is_isolated=True)

        writer.generate_tfrecords(self._train_data, 'train', 'Isolated Chalearn')
        writer.generate_tfrecords(self._val_data, 'val', 'Isolated Chalearn')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate TFRecords related to the Jester dataset')
    parser.add_argument('--root_dir', help='path to directory where "train" folder and "train_list.txt" file are stored', default='../../chalearn_isolated')
    parser.add_argument('--tfrecords_path', help='a path where TFRecords will be stored')
    args = parser.parse_args()

    provider = ChalearnIsolatedProvider(root_dir=args.root_dir, seq_h=100, seq_w=150, seq_l=60, batch_size=3)

    if args.tfrecords_path is not None:
        provider.generate_tfrecords(args.tfrecords_path)
