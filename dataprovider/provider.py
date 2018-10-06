import math
import os
from multiprocessing import cpu_count, Value, Lock, Process

import tensorflow as tf


class TFRecordWriter:
    """Class for utilizing data in the form of TFRecords"""

    def __init__(self, root_dir, record_length, seq_reading_func, is_isolated):
        """
        Initializes parameters for record writer
        :param root_dir: root directory for data, 'train' and 'validation' subdirs will be created there
        :param record_length: the number of video sequences in a single tf record
        :param seq_reading_func: a function for reading a single sequence, based on dataset. Must return a numpy tensor containing sequence, with first dimension
                                 as temporal dimension.
        :param is_isolated: True/False value, specifying wheter the dataset uses isolated sequences. In such case, a single scalar class_id will be assigned to
                                 each example. Otherwise, a list of class_ids is expected during tfrecord generation.
        """
        self.is_isolated = is_isolated
        self.seq_reading_func = seq_reading_func
        self.record_length = record_length
        self.root_dir = root_dir

    def _bytes_feature(self, value):
        """
        Converts value into Tensorflow ByteFeature
        """
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

    def _int64_feature(self, value):
        """
        Converts value into a single scalar - Tensorflow Int64Feature
        """
        return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

    def _int64_features(self, values):
        """
        Converts value into a list - Tensorflow Int64Feature
        """
        return tf.train.Feature(int64_list=tf.train.Int64List(value=values))

    def _split_data_into_chunks(self, data, num_processes):
        """
        Computes the number of tfrecords needed to save all data, then converts this number into list of tfrecord ids: [0, 1, 2, ... total tfrecords.
        Then this list is splitted into a 'num_process' chunks, each chunk for the single process.
        :param data: a list containing data
        :param num_processes: number of processes to run simultaneously to speed up work. Default: number of CPU cores
        :return: a nested list of data chunks, total number of tf_records
        """
        # Based on the length of data and the length of single tfrecord, compute total number of tf_records
        data_length = 2000  # len(data) # todo
        total_tfrecords = math.ceil(data_length / self.record_length)

        # create a list of tfrecord ids
        tfrecord_id_list = range(total_tfrecords)

        # an upper bound of chunk length (the last one will be probably shorter)
        chunk_length = math.ceil(total_tfrecords / num_processes)

        # split the list of tfrecord ids into chunk of chunk_length size
        chunks = [tfrecord_id_list[i * chunk_length: (i + 1) * chunk_length] for i in range(num_processes)]

        return chunks, total_tfrecords

    def write_tfrecord(self, data, tfrecord_ids, record_counter, lock, parent_path, total_tfrecords, dataset_name, suffix):
        """
        Writes a chunk of data into a number of tfrecords.
        :param data: list of pairs (sequence tensor, class_id/class_ids)
        :param tfrecord_ids: a list of tfrecord ids, computed by self.
        :param record_counter: a counter associated with number of tfrecords processed so far
        :param lock: lock related to multiprocessing, to avoid counter lock
        :param parent_path: the path to the main directory where records are stored
        :param total_tfrecords: total number of tfrecords to be made
        :param suffix: root path to the directory where datasets will be stored
        :param dataset_name: name of the dataset, only for printing purposes
        """
        data_length = len(data)
        for i in tfrecord_ids:
            # increment tfrecord counter
            with lock:
                record_counter.value += 1
            print('Generating TFRecords: {} of {}, dataset: {}, type: {}'.format(record_counter.value, total_tfrecords, dataset_name, suffix.upper()))

            # find data range for current tfrecord id
            current_data = data[i * self.record_length: (i + 1) * self.record_length]
            current_tfrecord_name = parent_path + '/{}_{}.tfrecord'.format(i * self.record_length, min((i + 1) * self.record_length, data_length))

            # itarate over data in single chunk
            with tf.python_io.TFRecordWriter(current_tfrecord_name) as writer:
                for name, class_id in current_data:
                    sequence_tensor = self.seq_reading_func(name)

                    # these data will be needed during data reading in order to apply proper reshaping
                    length = sequence_tensor.shape[0]
                    height = sequence_tensor.shape[1]
                    width = sequence_tensor.shape[2]

                    # write data to tf record file
                    sequence_bytes = sequence_tensor.tostring()  # code sequence tensor as bytes
                    example = tf.train.Example(features=tf.train.Features(feature={
                        'length': self._int64_feature(length),
                        'height': self._int64_feature(height),
                        'width': self._int64_feature(width),
                        'sequence_bytes': self._bytes_feature(sequence_bytes),
                        'class_id': self._int64_feature(class_id) if self.is_isolated else self._int64_features(class_id)}))

                    writer.write(example.SerializeToString())

    def generate_tfrecords(self, data, suffix, dataset_name, num_parallel_runs=None):
        """
        Creates train and val datasets in the format of TF records.
        :param data: list of pairs (dir_name/sequence_name, class_id/class_ids)
        :param suffix: root path to the directory where datasets will be stored
        :param dataset_name: name of the dataset, only for printing purposes
        :param num_parallel_runs: number of processes to run simultaneously to speed up work. Default: number of CPU cores
        """
        # make train and val directories
        parent_path = os.path.join(self.root_dir, suffix)
        os.makedirs(parent_path, exist_ok=True)

        # compute chunks of datas
        num_processes = num_parallel_runs if num_parallel_runs is not None else cpu_count()
        tfrecord_chunks, total_tfrecords = self._split_data_into_chunks(data, num_processes)

        # run processing on multiple processes
        record_counter = Value('i', 0)  # number of tfrecords processed so far
        lock = Lock()
        processes = []
        for chunk in tfrecord_chunks:
            proc = Process(target=self.write_tfrecord, args=(data, chunk, record_counter, lock, parent_path, total_tfrecords, dataset_name, suffix))
            processes.append(proc)
            proc.start()
        for proc in processes: proc.join()


class IsolatedSequenceProvider:
    """Class for providing data for isolated sequences (not continuous)"""

    def __init__(self, seq_h, seq_w, seq_l):
        """
        Initializes DataProvider
        :param seq_h: the height the images in sequence will be resized to
        :param seq_w: the width the images in sequence will be resized to
        :param seq_l: the length the sequence will be scaled to
        """
        self.seq_h = seq_h
        self.seq_w = seq_w
        self.seq_l = seq_l

    def _match_names_labels(self, **kwargs):
        """
        Matches filenames with corresponding labels
        :return: list(filenames, labels) for training set; list(filenames, labels) for validation set
        """
        raise NotImplementedError

    def _define_dataset(self, data):
        """
        Defines TF Dataset API for given data
        """
        raise NotImplementedError

    def dataset_handles(self):
        """
        Creates and returns dataset handles.
        """
        raise NotImplementedError
