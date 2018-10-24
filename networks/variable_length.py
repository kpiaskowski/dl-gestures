import tensorflow as tf
from tensorflow.contrib.layers import xavier_initializer, l2_regularizer
from tensorflow.contrib.rnn import OutputProjectionWrapper


class SimpleLSTMNet:
    """
    Simple 3D LSTM network. No attention, no residual connections.
    The LSTM part accepts blocks of data of shape [batch, time, depth, height, width, channels]
    """

    def __init__(self, inputs, num_classes, training_placeholder, labels=None, transpose_to_NDHWC=False):
        """
        Defines network
        :param sequence_length: vector of lengths of sequences in batch (specify only for online inference). If None (for example during training), the lengths of sequences will be inferred
        :param training_placeholder: a bool placeholder
        :param transpose_to_NDHWC: specifies whether inputs should be transposed from NDHWC to NCDHW
        :param inputs: input tensor of shape [batch, sequence length, sequence height, sequence width, number of channels]
        :param num_classes: number of neurons in last layer, associated with number of possible classes (including one `blank`)
        :param labels: labels - `None` for inference mode
        """
        labels = labels if labels is not None else [0]
        inputs = self._NDHWC_to_NCDHW(inputs) if transpose_to_NDHWC else inputs
        seq_lengths = self.find_sequence_lengths(labels, num_classes)

        self._data_format = 'channels_first' if transpose_to_NDHWC else 'channels_last'

        conv_tower = self._convolutional_tower(inputs, training_placeholder)
        self._logits = self._rnn(conv_tower, num_classes, training_placeholder, seq_lengths, hidden_size=2048, num_layers=2)

        # dummy label in situation when no label is provided
        self._loss = self._compute_loss(self._logits, labels, num_classes)
        self._accuracy = self._compute_accuracy(self._logits, labels)

    def find_sequence_lengths(self, labels, num_classes):
        """
        Finds the lengths of sequences (assuming that sequence ends with a special `BLANK` token, of index equal to number of classes in the dataset
        """
        # finds all `blank` items in labels (items with class_id==number of classes are considered `blanks`)
        blanks = tf.to_int32(tf.equal(labels, num_classes - 1))
        first_blank = tf.argmax(blanks, axis=-1, output_type=tf.int32)

        # If there is no `blank` in sequence (its length is greater/equal to global seq_len), then its `first_blank` value will be equal to 0. Let's create a vector stating, which
        # sequence in batch has such great length. Then, set their lengths to maximum sequence length
        long_sequences = tf.to_int32(tf.equal(first_blank, 0))
        max_sequence_length = tf.shape(labels)[1]
        long_sequences *= max_sequence_length

        # combine information of sequences with maximum and normal length
        lengths = long_sequences + first_blank
        return lengths

    def predictions(self):
        """Returns softmax predictions for each timestep"""
        return self._logits

    def metrics(self):
        """
        :return: loss, accuracy
        """
        return self._loss, self._accuracy

    def _NDHWC_to_NCDHW(self, input):
        """
        Changes data from NDHCW format to NCDHW
        :param input: tensor of shape [batch, length, height, width, channels]
        :return: tensor of shape [batch, channels, length, height, width]
        """
        return tf.transpose(input, [0, 4, 1, 2, 3])

    def _convolutional_tower(self, inputs, training_placeholder, activation_function=tf.nn.relu, L2_scale=0.001):
        """
        Defines convolutional tower.
        :param inputs: input tensor of shape [batch, sequence length, sequence height, sequence width, number of channels]
        :param training_placeholder: a placeholder
        :param L2_scale: value for L2 regularization
        :return: a tensor, result of a sequence of convolutions
        """
        with tf.variable_scope('convolutional_tower', initializer=xavier_initializer(), regularizer=l2_regularizer(L2_scale)):
            conv = tf.layers.conv3d(inputs, filters=24, kernel_size=3, strides=[1, 2, 2], padding='same', data_format=self._data_format)
            conv = tf.layers.batch_normalization(conv, training=training_placeholder, fused=True)
            conv = activation_function(conv)
            conv = tf.layers.max_pooling3d(conv, (1, 2, 2), (1, 2, 2), data_format=self._data_format)

            conv = tf.layers.conv3d(conv, filters=32, kernel_size=3, padding='same', data_format=self._data_format)
            conv = tf.layers.batch_normalization(conv, training=training_placeholder, fused=True)
            conv = activation_function(conv)
            conv = tf.layers.max_pooling3d(conv, (1, 2, 2), (1, 2, 2), data_format=self._data_format)

            conv = tf.layers.conv3d(conv, filters=64, kernel_size=3, padding='same', data_format=self._data_format)
            conv = tf.layers.batch_normalization(conv, training=training_placeholder, fused=True)
            conv = activation_function(conv)
            conv = tf.layers.conv3d(conv, filters=64, kernel_size=3, padding='same', data_format=self._data_format)
            conv = tf.layers.batch_normalization(conv, training=training_placeholder, fused=True)
            conv = activation_function(conv)
            conv = tf.layers.max_pooling3d(conv, (1, 2, 2), (1, 2, 2), data_format=self._data_format)

            conv = tf.layers.conv3d(conv, filters=128, kernel_size=3, padding='same', data_format=self._data_format)
            conv = tf.layers.batch_normalization(conv, training=training_placeholder, fused=True)
            conv = activation_function(conv)
            conv = tf.layers.conv3d(conv, filters=128, kernel_size=3, padding='same', data_format=self._data_format)
            conv = tf.layers.batch_normalization(conv, training=training_placeholder, fused=True)
            conv = activation_function(conv)
            conv = tf.layers.max_pooling3d(conv, (1, 2, 2), (1, 2, 2), data_format=self._data_format)

            conv = tf.layers.conv3d(conv, filters=256, kernel_size=3, padding='same', data_format=self._data_format)
            conv = tf.layers.batch_normalization(conv, training=training_placeholder, fused=True)
            conv = activation_function(conv)
            conv = tf.layers.conv3d(conv, filters=256, kernel_size=3, padding='same', data_format=self._data_format)
            conv = tf.layers.batch_normalization(conv, training=training_placeholder, fused=True)
            conv = activation_function(conv)
            conv = tf.layers.max_pooling3d(conv, (1, 2, 2), (1, 2, 2), data_format=self._data_format)

            conv = tf.layers.conv3d(conv, filters=512, kernel_size=3, padding='same', data_format=self._data_format)
            conv = tf.layers.batch_normalization(conv, training=training_placeholder, fused=True)
            conv = activation_function(conv)
            conv = tf.layers.conv3d(conv, filters=512, kernel_size=3, padding='same', data_format=self._data_format)
            conv = tf.layers.batch_normalization(conv, training=training_placeholder, fused=True)
            conv = activation_function(conv)

            conv = tf.layers.max_pooling3d(conv, (1, 2, 2), (1, 2, 2), data_format=self._data_format)

            return conv

    def _rnn(self, conv_inputs, output_size, training_placeholder, sequence_lengths, hidden_size, num_layers=4):
        """
        Takes output of convolutional tower and passes it through a multilayer LSTM
        :param sequence_lengths: vector containing lengths of sequences in batch
        :param num_layers: number of layers in multilayer RNN (n-1 normal layers and one output layer with projection function)
        :param conv_inputs: NOT reshaped output of convolutional tower (shaped [batch, l, h, w, channels]
        :param output_size: number of neurons in last layer, associated with number of possible class_ids
        :param training_placeholder: a placeholder
        :return: a tensor [batch_size, timesteps, num_classes] as an output of each timmestep of LSTM
        """
        keep_prob = tf.maximum(1 - tf.to_float(training_placeholder), 0.85)  # during training, keep_prob=0.85, during validation keep_prob=1.0
        with tf.variable_scope('sequential', initializer=xavier_initializer()):
            # flatten h, w, c dimensions
            flattened = tf.squeeze(tf.squeeze(conv_inputs, -2), -2)

            # only forward LSTMs are used
            fw_cells = []
            for i in range(num_layers):
                fw_cell = tf.nn.rnn_cell.BasicLSTMCell(hidden_size)
                if i < num_layers - 1:
                    fw_cell = tf.nn.rnn_cell.DropoutWrapper(fw_cell, output_keep_prob=keep_prob)
                if i == num_layers - 1:
                    fw_cell = OutputProjectionWrapper(fw_cell, output_size, tf.nn.tanh)

                fw_cells.append(fw_cell)

            fw_cells = tf.nn.rnn_cell.MultiRNNCell(fw_cells)
            outputs, _ = tf.nn.dynamic_rnn(cell=fw_cells,
                                           inputs=flattened,
                                           sequence_length=sequence_lengths,
                                           dtype=tf.float32)

            return outputs

    def _compute_loss(self, logits, labels, num_classes):
        """
        Specifies loss function.
        :param num_classes: number of classes
        :param labels: vector of labels, one label for each timestep
        :param logits: outputs from the network (without softmax!)
        :return: output of loss function
        """
        # During loss computation, do not use elements that are beyond the length of the sequence
        loss_weights = tf.to_int32(tf.not_equal(labels, num_classes - 1))

        # convert labels to one hot vectors (there is something strange with sparse_softmax_cross entropy, causing NANS)
        one_hot = tf.one_hot(labels, num_classes)
        loss = tf.losses.softmax_cross_entropy(one_hot, logits, loss_weights)
        return loss

    def _compute_accuracy(self, logits, labels):
        """
        Computes accuracy.
        :param logits: outputs from the network
        :param labels: a label, specified as a single scalar - class_id (NOT one hot)
        :return: output of accuracy function
        """
        return tf.reduce_mean(tf.to_float(tf.equal(labels, tf.argmax(logits, -1, output_type=tf.int32))))
