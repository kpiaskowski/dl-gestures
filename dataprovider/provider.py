import math
import os
from multiprocessing import cpu_count, Value, Lock, Process

import tensorflow as tf
from tensorflow.python.ops.image_ops_impl import ResizeMethod


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
        data_length = len(data)
        total_tfrecords = math.ceil(data_length / self.record_length)

        # create a list of tfrecord ids
        tfrecord_id_list = range(total_tfrecords)

        # an upper bound of chunk length (the last one will be probably shorter)
        chunk_length = math.ceil(total_tfrecords / num_processes)

        # split the list of tfrecord ids into chunk of chunk_length size
        chunks = [tfrecord_id_list[i * chunk_length: (i + 1) * chunk_length] for i in range(num_processes)]

        return chunks, total_tfrecords

    def _print_processing_stats(self, processed_items, total_items, dataset_name, suffix):
        """
        Prints statistics related to generating data.
        :param processed_items: number of items processed so far
        :param total_items: total number of items
        :param suffix: root path to the directory where datasets will be stored
        :param dataset_name: name of the dataset, only for printing purposes
        """
        print('\rGenerating TFRecords: {} of {}, dataset: {}, type: {}'.format(processed_items, total_items, dataset_name, suffix.upper()), end='')

    def write_tfrecord(self, data, tfrecord_ids, record_counter, lock, parent_path, total_tfrecords, dataset_name, suffix, existing_records):
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
        :param existing_records: a list of existing tfrecord names (might be empty)
        """
        data_length = len(data)
        for i in tfrecord_ids:
            # increment tfrecord counter
            with lock:
                record_counter.value += 1
            self._print_processing_stats(record_counter.value, total_tfrecords, dataset_name, suffix)

            # find data range for current tfrecord id
            current_data = data[i * self.record_length: (i + 1) * self.record_length]
            current_tfrecord_name = parent_path + '/{}_{}.tfrecord'.format(i * self.record_length, min((i + 1) * self.record_length, data_length))

            # if there is already a record with that name, assume that the saved one is correct and move to next chunk of data
            if current_tfrecord_name in existing_records:
                continue

            # itarate over data in single chunk
            with tf.python_io.TFRecordWriter(current_tfrecord_name) as writer:
                for name, class_id in current_data:
                    try:
                        sequence_tensor = self.seq_reading_func(name)
                    except Exception as e:
                        print(e)
                        continue

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

    def _find_existinig_records(self, path):
        """
        Checks if any records exist in the given path and asserts that their lengths match the length of currently processed tfrecords (otherwise exception
        is raised)
        :param path: path to the folder with tfrecords
        :return: a list of existing tfrecords (might be empty)
        """
        existing_records = [os.path.join(path, tfrecord) for tfrecord in os.listdir(path)]

        if existing_records:
            sample_record = existing_records[0].split('/')[-1].rstrip('.tfrecord')  # get only record range
            record_length = int(sample_record.split('_')[1]) - int(sample_record.split('_')[0])
            if record_length != self.record_length:
                raise Exception('The length of saved tfrecords does not match the length you specified now! Please change the length or remove dataset')

        return existing_records

    def generate_tfrecords(self, data, suffix, dataset_name, num_parallel_runs=None):
        """
        Creates train and val datasets in the format of TF records.
        Also tries to continue generating in case of interruption. Please note, however, that this function cannot ensure integrity of tfrecords being processed
        during interruption. You need to remove such records by hand (they might be recognized by their abnormally small size, for example).
        :param data: list of pairs (dir_name/sequence_name, class_id/class_ids)
        :param suffix: root path to the directory where datasets will be stored, must be either 'train' or 'val'
        :param dataset_name: name of the dataset, only for printing purposes
        :param num_parallel_runs: number of processes to run simultaneously to speed up work. Default: number of CPU cores
        """
        if not (suffix == 'train' or suffix == 'val'):
            raise Exception('Suffix must be either "train" or "val"!')

        # make train and val directories
        parent_path = os.path.join(self.root_dir, suffix)
        os.makedirs(parent_path, exist_ok=True)

        # check if there are any existing tfrecords
        existing_tfrecords = self._find_existinig_records(parent_path)

        # compute chunks of datas
        num_processes = num_parallel_runs if num_parallel_runs is not None else cpu_count()
        tfrecord_chunks, total_tfrecords = self._split_data_into_chunks(data, num_processes)

        # run processing on multiple processes
        record_counter = Value('i', 0)  # number of tfrecords processed so far
        lock = Lock()
        processes = []
        for chunk in tfrecord_chunks:
            proc = Process(target=self.write_tfrecord,
                           args=(data, chunk, record_counter, lock, parent_path, total_tfrecords, dataset_name, suffix, existing_tfrecords))
            processes.append(proc)
            proc.start()
        for proc in processes:
            proc.join()
        print()


class IsolatedSequenceProvider:
    """Class for providing data for isolated sequences (not continuous)"""

    def __init__(self, seq_h, seq_w, seq_l, batch_size, fake_continuous=False):
        """
        Initializes DataProvider
        :param fake_continuous: True/False - specifies, wheter the dataset should imitate a continuous data, by tiling class_id of sequence to match sequence length
        :param seq_h: the height the images in sequence will be resized to (online, during training)
        :param seq_w: the width the images in sequence will be resized to (online, during training)
        :param seq_l: the length the sequence will be scaled to (online, during training)
        :param batch_size: size of batch
        """
        self.seq_h = seq_h
        self.seq_w = seq_w
        self.seq_l = seq_l
        self.batch_size = batch_size

        # params for random cropping
        self.min_side_ratio = 0.9  # minimal lengths of sides of cropped subtensors are at least 90% of original sides
        self.min_length_ratio = 0.8  # minimum percentage length of temporally stretched sequence
        self.max_length_ratio = 1.2  # maximum percentage length of temporally stretched sequence

        # params that need to be computed in child classes
        self._train_data = None
        self._val_data = None
        self._num_classes = None

        # processing functions that depend on fake_continuous
        self._fake_continuous = fake_continuous
        self._maybe_expand_classes = self._fake_multiple_classes if self._fake_continuous else lambda x, y: (x, y)
        self._pad = self._pad_all_data if self._fake_continuous else self._pad_sequnce_only
        self._fix_class_shape = self._fix_continuous_class_shape if self._fake_continuous else self._fix_single_class_shape

    def num_classes(self):
        """
        Returns number of classes in dataset, depending on wheter it is treated as isolated or pseudocontinuous dataset. In the latter case, an artificial `BLANK` class is added
        (which is treated as a padding)
        """
        return self._num_classes + 1 if self._fake_continuous else self._num_classes

    def _match_names_labels(self, **kwargs):
        """
        Matches filenames with corresponding labels
        :return: list(filenames, labels) for training set; list(filenames, labels) for validation set
        """
        raise NotImplementedError

    def _fake_multiple_classes(self, sequence_tensor, class_id):
        """
        Tiles single class to create a vector [class_id, class_id, class_id...class_id] of length equal to sequence length
        :param class_id: class_id of the sequence
        :return: unchanged sequence_tensor, a class_id as a tile vector
        """
        length = tf.shape(sequence_tensor)[0]
        return sequence_tensor, tf.ones(length, tf.int32) * class_id

    def _parse_tfrecord(self, example_proto):
        """
        Parses and decodes a single example from tfrecord
        :param example_proto: tfrecord from TFRecordDataset
        :return: sequence tensor, class_id
        """
        features = {"sequence_bytes": tf.FixedLenFeature([], tf.string),
                    "length": tf.FixedLenFeature([], tf.int64),
                    "width": tf.FixedLenFeature([], tf.int64),
                    "class_id": tf.FixedLenFeature([], tf.int64),
                    "height": tf.FixedLenFeature([], tf.int64)}
        parsed_features = tf.parse_single_example(example_proto, features)

        # load and set shape to sequence tensor
        length = tf.cast(parsed_features['length'], tf.int32)
        height = tf.cast(parsed_features['height'], tf.int32)
        width = tf.cast(parsed_features['width'], tf.int32)
        raw_sequence = tf.decode_raw(parsed_features['sequence_bytes'], tf.uint8)

        sequence_shape = tf.stack([length, height, width, 3])
        sequence_tensor = tf.reshape(raw_sequence, sequence_shape)

        # parse class_id
        class_id = tf.cast(parsed_features['class_id'], tf.int32)
        return sequence_tensor, class_id

    def _resize_spatially(self, sequence_tensor, class_id):
        """
        Resizes sequence tensor in spatial dimensions (h, w), using bilinear interpolation
        :return: sequence resized spatially, unchanged class_id
        """
        resized_sequence = tf.image.resize_images(sequence_tensor, size=(self.seq_h, self.seq_w))
        return resized_sequence, class_id

    def _stretch_temporally(self, sequence_tensor, class_id):
        """
        Stretches sequence temporally in range 0.8...1.2, using nearest neighbor interpolation. Sequence is described as a tensor of shape [l, h, w, c]. Function
        tf.image.resize_images works only on dimensions h and w, so we need to permute the order of dimensions to make a hackish resizing, then revert the
        operation to get the initial sequence. Note, that both h and w, are constant (resized earlier to self.h, self.w), so we know them beforehand and might
        use a hack of resizing w to the same, identical w.
        :return: sequence resized temporally, unchanged class_id
        """
        # find new length
        l = tf.to_float(tf.shape(sequence_tensor)[0])
        min_l = tf.cast(self.min_length_ratio * l, tf.int32)
        max_l = tf.cast(self.max_length_ratio * l, tf.int32)
        new_l = tf.random_uniform([1], min_l, max_l, tf.int32)[0]  # rando_uniform requires 1D tensor as shape, so we extract only 0th index

        # swap [l, h, w, c] to  [h, l, w, c]
        sequence = tf.transpose(sequence_tensor, perm=[1, 0, 2, 3])
        sequence = tf.image.resize_images(sequence, size=(new_l, self.seq_w), method=ResizeMethod.NEAREST_NEIGHBOR)

        # swap back to [l, h, w, c]
        sequence = tf.transpose(sequence, perm=[1, 0, 2, 3])
        return sequence, class_id

    def _standardize(self, sequence_tensor, class_id):
        """
        Scales image to <0...1>, then standardizes it. Standardization is based on tf.image.per_image_standardization (but allows tensors of arbitrary shape),
        so it should be numerically safe.
        :return: standardized sequence, unchanged class_id
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

        return standardized_sequence, class_id

    def _pad_all_data(self, sequence_tensor, class_id):
        """
        Pads sequence and class_ids (as vector, in case of pseudo continuous data), as opposite to temporal stretching. If data is longer than desired length, it cuts it.
        Video sequence is padded with zeros, class ids with -1 (because there is a possibility that class_id will be equal to 0
        :return: standardized sequence, unchanged class_id
        """
        shape = tf.shape(sequence_tensor)
        l = shape[0]
        h = shape[1]
        w = shape[2]
        c = shape[3]

        # tf hack to avoid using ifs - note that remainder might be of length = 0
        remainder = tf.maximum(self.seq_l - l, 0)
        seq_padding = tf.zeros([remainder, h, w, c], tf.float32)
        sequence = tf.concat([sequence_tensor, seq_padding], axis=0)
        cls_padding = tf.ones([remainder], tf.int32) * self._num_classes
        class_ids = tf.concat([class_id, cls_padding], 0)

        # slice sequence (again tf hack, sequence after slicing might not change at all)
        sequence = sequence[:self.seq_l, :, :, :]
        class_ids = class_ids[:self.seq_l]
        return sequence, class_ids

    def _pad_sequnce_only(self, sequence_tensor, class_id):
        """
        Pads sequence with zeros (as opposite to temporal stretching). If sequence is longer than desired length, it cuts it.
        :return: standardized sequence, unchanged class_id
        """
        shape = tf.shape(sequence_tensor)
        l = shape[0]
        h = shape[1]
        w = shape[2]
        c = shape[3]

        # tf hack to avoid using ifs - note that remainder might be of length = 0
        remainder = tf.maximum(self.seq_l - l, 0)
        padding = tf.zeros([remainder, h, w, c], tf.float32)
        sequence = tf.concat([sequence_tensor, padding], axis=0)

        # slice sequence (again tf hack, sequence after slicing might not change at all)
        sequence = sequence[:self.seq_l, :, :, :]
        return sequence, class_id

    def _random_contrast_change(self, sequence_tensor, class_id):
        """
        Apply random contrast changes, by pixelwise multiplication of sequence and random noise in range(0.9...1.1)
        :return: noisy sequence, unchanged class_id
        """
        noise = tf.random_uniform(tf.shape(sequence_tensor), minval=0.9, maxval=1.1)
        noisy_sequence = tf.multiply(sequence_tensor, noise)
        return noisy_sequence, class_id

    def _random_brightness_change(self, sequence_tensor, class_id):
        """
        Apply random brightnesschanges, by pixelwise addition of sequence and random noise in range(-0.1...0.1)
        :return: noisy sequence, unchanged class_id
        """
        noise = tf.random_uniform(tf.shape(sequence_tensor), minval=-0.1, maxval=0.1)
        noisy_sequence = tf.add(sequence_tensor, noise)
        return noisy_sequence, class_id

    def _random_spatial_crop(self, sequence_tensor, class_id):
        """
        Apply random cropping in spatial dimension.
        :return: sequence cropped spatially, unchanged class_id
        """
        shape = tf.shape(sequence_tensor)
        l = shape[0]
        h = shape[1]
        w = shape[2]
        c = shape[3]
        min_h = tf.cast(self.min_side_ratio * tf.to_float(h), tf.int32)
        min_w = tf.cast(self.min_side_ratio * tf.to_float(w), tf.int32)

        # generate new, random dimensions of sequence tensor. [0] is for extracting only a single scalar. It is needed, because random uniform requires 1D tensor as shape
        new_h = tf.random_uniform(shape=[1], minval=min_h, maxval=h, dtype=tf.int32)[0]
        new_w = tf.random_uniform(shape=[1], minval=min_w, maxval=w, dtype=tf.int32)[0]

        # apply cropping
        cropped_sequence = tf.random_crop(sequence_tensor, [l, new_h, new_w, c])
        return cropped_sequence, class_id

    def _random_temporal_crop(self, sequence_tensor, class_id):
        """
        Apply random cropping in temporal dimension.
        :return: sequence cropped spatially, unchanged class_id
        """
        shape = tf.shape(sequence_tensor)
        l = shape[0]
        h = shape[1]
        w = shape[2]
        c = shape[3]
        min_l = tf.cast(self.min_side_ratio * tf.to_float(l), tf.int32)

        # generate new, random dimension of sequence tensor. [0] is for extracting only a single scalar. It is needed, because random uniform requires 1D tensor as shape
        new_l = tf.random_uniform(shape=[1], minval=min_l, maxval=l, dtype=tf.int32)[0]

        # apply cropping
        cropped_sequence = tf.random_crop(sequence_tensor, [new_l, h, w, c])
        return cropped_sequence, class_id

    def _define_dataset_pipeline(self, tfrecord_paths):
        """
        Defines TF Dataset API for given data
        :param tfrecord_paths: paths of tfrecords
        """
        num_parallel = cpu_count()

        # define datast pipeline
        dataset = tf.data.TFRecordDataset(tfrecord_paths)
        dataset = dataset.shuffle(300)
        dataset = dataset.map(self._parse_tfrecord, num_parallel_calls=num_parallel)
        dataset = dataset.map(self._standardize, num_parallel_calls=num_parallel)

        # apply random perturbations
        dataset = dataset.map(self._random_contrast_change, num_parallel_calls=num_parallel)
        dataset = dataset.map(self._random_brightness_change, num_parallel_calls=num_parallel)

        # random cropping and stretching
        dataset = dataset.map(self._random_spatial_crop, num_parallel_calls=num_parallel)
        dataset = dataset.map(self._random_temporal_crop, num_parallel_calls=num_parallel)
        dataset = dataset.map(self._stretch_temporally, num_parallel_calls=num_parallel)

        # treat class scalar as vector, if pseudo continuity is set
        dataset = dataset.map(self._maybe_expand_classes, num_parallel_calls=num_parallel)

        # resize to fixed dimensions
        dataset = dataset.map(self._resize_spatially, num_parallel_calls=num_parallel)
        dataset = dataset.map(self._pad, num_parallel_calls=num_parallel)

        dataset = dataset.shuffle(10)
        dataset = dataset.prefetch(self.batch_size)
        dataset = dataset.batch(self.batch_size)
        dataset = dataset.repeat()
        return dataset

    def _fix_single_class_shape(self, class_tensor, batch_size):
        """
        Fixes shape of classes (provided as a single scalar)
        """
        return tf.reshape(class_tensor, [batch_size])

    def _fix_shape(self, batch_size, sequence_tensor, class_tensor):
        """
        Does final reshape of tensors (it is needed because information about dimension is somehow not passed in dataset api.
        It simply sets identical shape to both tensors.
        """
        class_id = self._fix_class_shape(class_tensor, batch_size)
        sequence_tensor = tf.reshape(sequence_tensor, [batch_size, self.seq_l, self.seq_h, self.seq_w, 3])
        return sequence_tensor, class_id

    def _fix_continuous_class_shape(self, class_tensor, batch_size):
        """
        Fixes shape of classes (provided as a vector of scalars)
        """
        return tf.reshape(class_tensor, [batch_size, self.seq_l])

    def create_dataset_handles(self, root_dir):
        """
        Creates and returns TF Dataset API handles.
        :param root_dir: root directory of the dataset, containing 2 subfolders: 'train' and 'val', each containing multiple tfrecords
        :return: sequence_tensor, class_id, iterator, train_iterator, val_iterator, handle
        """
        # find paths to the tfrecords
        train_path = os.path.join(root_dir, 'train')
        val_path = os.path.join(root_dir, 'val')
        train_records = [os.path.join(train_path, record_name) for record_name in os.listdir(train_path)]
        val_records = [os.path.join(val_path, record_name) for record_name in os.listdir(val_path)]

        train_dataset = self._define_dataset_pipeline(train_records)
        val_dataset = self._define_dataset_pipeline(val_records)

        handle = tf.placeholder(tf.string, shape=[])
        iterator = tf.data.Iterator.from_string_handle(handle, train_dataset.output_types, train_dataset.output_shapes)
        sequence_tensor, class_id = iterator.get_next()

        # tensorflow has problems with passing info about dimensionality, so a manual intervention is needed
        batch_size = tf.shape(sequence_tensor)[0]
        sequence_tensor, class_id = self._fix_shape(batch_size, sequence_tensor, class_id)

        train_iterator = train_dataset.make_one_shot_iterator()
        val_iterator = val_dataset.make_one_shot_iterator()

        return sequence_tensor, class_id, iterator, train_iterator, val_iterator, handle

    def _read_sequence(self, path):
        """
        Reads content of the sequence under given path and converts it to sequence tensor
        """
        raise NotImplementedError

    def generate_tfrecords(self, root_dir, dataset_name):
        """
        # Generates data in the form of TF records
        # :param dataset_name: name of the dataset, just for printing purposes
        # :param root_dir: root directory, where 'train' and 'validation' data will be stored.
        """
        # 100 sequences per single TFRecord
        writer = TFRecordWriter(root_dir, record_length=50, seq_reading_func=self._read_sequence, is_isolated=True)

        writer.generate_tfrecords(self._train_data, 'train', dataset_name)
        writer.generate_tfrecords(self._val_data, 'val', dataset_name)
