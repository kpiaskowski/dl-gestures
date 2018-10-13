import tensorflow as tf
from tensorflow.contrib.layers import xavier_initializer, l2_regularizer


class SimpleConvNet:
    """
    Simple 3D convolitional network. No LSTM (fixed length), no attention, no residual connections.
    The convolutional part preserves temporal dimensionality (output temporal dimension is the same as input temporal dimension)
    """

    def __init__(self, inputs, output_size, is_training, labels=None, transpose_to_NDHWC=False):
        """
        Defines network
        :param is_training: True/False value
        :param transpose_to_NDHWC: specifies whether inputs should be transposed from NDHWC to NCDHW
        :param inputs: input tensor of shape [batch, sequence length, sequence height, sequence width, number of channels]
        :param output_size: number of neurons in last layer, associated with number of possible class_ids
        :param labels: labels - `None` for inference mode
        """
        labels = labels if labels is not None else [0]
        inputs = self._NDHWC_to_NCDHW(inputs) if transpose_to_NDHWC else inputs

        self._data_format = 'channels_first' if transpose_to_NDHWC else 'channels_last'

        self._conv_tower = self._convolutional_tower(inputs, is_training)
        self._logits, self._softmax_outputs = self._final_dense(self._conv_tower, output_size, is_training)

        # dummy label in situation when no label is provided
        self._loss = self._compute_loss(self._logits, labels)
        self._accuracy = self._compute_accuracy(self._logits, labels)

    def predictions(self):
        """
        :return: softmax predictions
        """
        return self._softmax_outputs

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

    def _convolutional_tower(self, inputs, is_training, activation_function=tf.nn.relu, L2_scale=0.001):
        """
        Defines convolutional tower.
        :param inputs: input tensor of shape [batch, sequence length, sequence height, sequence width, number of channels]
        :param is_training: a placeholder
        :param L2_scale: value for L2 regularization
        :return: a tensor, result of a sequence of convolutions
        """
        with tf.variable_scope('convolutional_tower', initializer=xavier_initializer(), regularizer=l2_regularizer(L2_scale)):
            conv = tf.layers.conv3d(inputs, filters=24, kernel_size=3, strides=[1, 2, 2], padding='same', data_format=self._data_format)
            conv = tf.layers.batch_normalization(conv, training=is_training, fused=True)
            conv = activation_function(conv)
            conv = tf.layers.max_pooling3d(conv, (1, 2, 2), (1, 2, 2), data_format=self._data_format)

            conv = tf.layers.conv3d(conv, filters=32, kernel_size=3, padding='same', data_format=self._data_format)
            conv = tf.layers.batch_normalization(conv, training=is_training, fused=True)
            conv = activation_function(conv)
            conv = tf.layers.max_pooling3d(conv, (1, 2, 2), (1, 2, 2), data_format=self._data_format)

            conv = tf.layers.conv3d(conv, filters=64, kernel_size=3, padding='same', data_format=self._data_format)
            conv = tf.layers.batch_normalization(conv, training=is_training, fused=True)
            conv = activation_function(conv)
            conv = tf.layers.conv3d(conv, filters=64, kernel_size=3, padding='same', data_format=self._data_format)
            conv = tf.layers.batch_normalization(conv, training=is_training, fused=True)
            conv = activation_function(conv)
            conv = tf.layers.max_pooling3d(conv, (1, 2, 2), (1, 2, 2), data_format=self._data_format)

            conv = tf.layers.conv3d(conv, filters=128, kernel_size=3, padding='same', data_format=self._data_format)
            conv = tf.layers.batch_normalization(conv, training=is_training, fused=True)
            conv = activation_function(conv)
            conv = tf.layers.conv3d(conv, filters=128, kernel_size=3, padding='same', data_format=self._data_format)
            conv = tf.layers.batch_normalization(conv, training=is_training, fused=True)
            conv = activation_function(conv)
            conv = tf.layers.max_pooling3d(conv, (1, 2, 2), (1, 2, 2), data_format=self._data_format)

            conv = tf.layers.conv3d(conv, filters=256, kernel_size=3, padding='same', data_format=self._data_format)
            conv = tf.layers.batch_normalization(conv, training=is_training, fused=True)
            conv = activation_function(conv)
            conv = tf.layers.conv3d(conv, filters=256, kernel_size=3, padding='same', data_format=self._data_format)
            conv = tf.layers.batch_normalization(conv, training=is_training, fused=True)
            conv = activation_function(conv)
            conv = tf.layers.max_pooling3d(conv, (1, 2, 2), (1, 2, 2), data_format=self._data_format)

            conv = tf.layers.conv3d(conv, filters=512, kernel_size=3, padding='same', data_format=self._data_format)
            conv = tf.layers.batch_normalization(conv, training=is_training, fused=True)
            conv = activation_function(conv)
            conv = tf.layers.conv3d(conv, filters=512, kernel_size=3, padding='same', data_format=self._data_format)
            conv = tf.layers.batch_normalization(conv, training=is_training, fused=True)
            conv = activation_function(conv)
            conv = tf.layers.max_pooling3d(conv, (1, 2, 2), (1, 2, 2), data_format=self._data_format)

            return conv

    def _final_dense(self, conv_inputs, output_size, is_training, activation_function=tf.nn.relu, L2_scale=0.001):
        """
        Takes output of convolutional tower and passes it through a set of fully connected layers
        :param conv_inputs: NOT reshaped output of convolutional tower (shaped [[batch, l, h, w, channels]
        :param output_size: number of neurons in last layer, associated with number of possible class_ids
        :param is_training: a placeholder
        :param L2_scale: value for L2 regularization
        :return: two vectors of size `output_size`: one with predictions without softmax, second with softmax
        """
        with tf.variable_scope('dense', initializer=xavier_initializer(), regularizer=l2_regularizer(L2_scale)):
            dense = tf.layers.flatten(conv_inputs)
            dense = tf.layers.dense(dense, 2048)
            dense = tf.layers.batch_normalization(dense, training=is_training, fused=True)
            dense = activation_function(dense)
            dense = tf.layers.dropout(dense, training=is_training)

            dense = tf.layers.dense(dense, 2048)
            dense = tf.layers.batch_normalization(dense, training=is_training, fused=True)
            dense = activation_function(dense)
            dense = tf.layers.dropout(dense, training=is_training)

            raw_output = tf.layers.dense(dense, output_size, activation=None)
            softmax_output = tf.nn.softmax(raw_output, axis=-1)

            return raw_output, softmax_output

    def _compute_loss(self, logits, labels):
        """
        Specifies loss function.
        :param labels: a label, specified as a single scalar - class_id (NOT one hot)
        :param logits: outputs from the network (without softmax!)
        :return: output of loss function
        """
        return tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)

    def _compute_accuracy(self, logits, labels):
        """
        Computes accuracy.
        :param logits: outputs from the network
        :param labels: a label, specified as a single scalar - class_id (NOT one hot)
        :return: output of accuracy function
        """
        return tf.reduce_mean(tf.to_float(tf.equal(labels, tf.argmax(logits, -1, output_type=tf.int32))))
