import argparse
import os
import tensorflow as tf
import cv2
import numpy as np

if __name__ == '__main__':
    from provider import IsolatedSequenceProvider
else:
    from dataprovider.provider import IsolatedSequenceProvider


class JesterProvider(IsolatedSequenceProvider):
    def __init__(self, seq_h, seq_w, seq_l, batch_size, fake_continuous=False):
        """
        Initializes Jester dataset.
        :param seq_h: the height the images in sequence will be resized to
        :param seq_w: the width the images in sequence will be resized to
        :param seg_l: the length the sequence will be scaled to
        :param batch_size: size of batch
        :param fake_continuous: True/False - specifies, wheter the dataset should imitate a continuous data, by tiling class_id of sequence to match sequence length
        """
        super().__init__(seq_h, seq_w, seq_l, batch_size, fake_continuous)
        self.classes = ['Doing other things', 'Drumming Fingers', 'No gesture', 'Pulling Hand In',
                        'Pulling Two Fingers In', 'Pushing Hand Away', 'Pushing Two Fingers Away',
                        'Rolling Hand Backward', 'Rolling Hand Forward', 'Shaking Hand', 'Sliding Two Fingers Down',
                        'Sliding Two Fingers Left', 'Sliding Two Fingers Right',
                        'Sliding Two Fingers Up', 'Stop Sign', 'Swiping Down', 'Swiping Left', 'Swiping Right',
                        'Swiping Up', 'Thumb Down', 'Thumb Up', 'Turning Hand Clockwise',
                        'Turning Hand Counterclockwise', 'Zooming In With Full Hand', 'Zooming In With Two Fingers',
                        'Zooming Out With Full Hand', 'Zooming Out With Two Fingers']
        self._num_classes = 27

    def convert_to_tfrecords(self, data_dir, csv_dir, tfrecords_path):
        """
        Converts native data from dataset to TFRecords.
        :param data_dir: path to root directory with data
        :param csv_dir: path to root directory with csv_files
        :param tfrecords_path: path to the parent directory, where tfrecords will be stored
        """
        self._data_dir = data_dir

        # generate train and validation pairs of (image name, label)
        self._train_data = self._match_names_labels(csv_path=os.path.join(csv_dir, 'jester-v1-train.csv'))
        self._val_data = self._match_names_labels(csv_path=os.path.join(csv_dir, 'jester-v1-validation.csv'))

        # make correction for non-existing directories
        avl_dirs = set(os.listdir(data_dir))
        self._train_data = [n for n in self._train_data if n[0].split('/')[-1] in avl_dirs]
        self._val_data = [n for n in self._val_data if n[0].split('/')[-1] in avl_dirs]

        super().generate_tfrecords(tfrecords_path, 'Jester')

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
            class_id = self.classes.index(cls.rstrip('\n'))
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
    parser.add_argument('--data_dir',
                        help='path to the data where parent folder, where Jester folders with png images are stored',
                        default='../../jester/data')
    parser.add_argument('--csv_dir', help='path to the data where parent folder, where Jester csv files are stored',
                        default='../../jester/csv')
    parser.add_argument('--tfrecords_path', help='a path where TFRecords will be stored')
    args = parser.parse_args()

    provider = JesterProvider(seq_h=100, seq_w=150, seq_l=60, batch_size=3)

    # generate
    if all(arg is not None for arg in [args.tfrecords_path, args.csv_dir, args.data_dir]):
        provider.convert_to_tfrecords(data_dir=args.data_dir, csv_dir=args.csv_dir, tfrecords_path=args.tfrecords_path)

    # # uncomment to check how to fetch data from dataset (and verify wheter it works)
    # data_dir = 'jester_test'  # path to where tfrecords are stored
    # sequence_tensor, class_id, iterator, train_iterator, val_iterator, handle = provider.create_dataset_handles(root_dir=data_dir)
    #
    # with tf.Session() as sess:
    #     # initialize datasets
    #     train_handle = sess.run(train_iterator.string_handle())
    #     val_handle = sess.run(val_iterator.string_handle())
    #
    #     # train
    #     for i in range(5):
    #         seq, cls = sess.run([sequence_tensor, class_id], feed_dict={handle: train_handle})
    #         print('train', i, seq.shape, cls)
    #     # val
    #     for i in range(5):
    #         seq, cls = sess.run([sequence_tensor, class_id], feed_dict={handle: val_handle})
    #         print('validation', i, seq.shape, cls)
