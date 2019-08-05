import tensorflow as tf

class resnet():

    def __init__(self, num_classes, is_training=True):
        self.is_training = is_training
        self.num_classes = num_classes
        self.layers = []

    def _conv_layer(self, net, filters, activation=None, name='Conv'):
        return tf.layers.conv2d(net, filters=filters, kernel_size=[3, 3], strides=[1, 1], padding='same',
                                kernel_initializer=tf.truncated_normal_initializer(stddev=0.1),
                                activation=activation,
                                name=name)

    def _conv_bn_relu_layer(self, net, filters, stride=1, name='Conv_bn_relu'):

        net = tf.layers.conv2d(
            net, filters=filters, kernel_size=[3, 3],
            strides=[stride, stride], padding='same', name=name)
        net = self._bn_layer(net, self.is_training)
        net = tf.nn.relu(net)

        return net

    def _bn_layer(self, net, training, name='BN'):
        return tf.layers.batch_normalization(
            net, training=training, name=name)

    def _max_pool(self, net, name='Max_Pool'):
        return tf.layers.max_pooling2d(net, pool_size=[2, 2], strides=[2, 2], padding='same', name=name)

    def _avg_pool(self, net, name='Avg_Pool'):
        return tf.layers.average_pooling2d(net, pool_size=[2, 2], strides=[2, 2], padding='valid', name=name)

    def _fc_layer(self, net, num_classes, name='FC'):
        return tf.layers.dense(net, num_classes, name=name)

    def _residual_block(self, input_layer, output_channel, first_block=False):
        '''
        Defines a residual block in ResNet
        :param input_layer: 4D tensor
        :param output_channel: int. return_tensor.get_shape().as_list()[-1] = output_channel
        :param first_block: if this is the first residual block of the whole network
        :return: 4D tensor.
        '''
        input_channel = input_layer.get_shape().as_list()[-1]

        # When it's time to "shrink" the image size, we use stride = 2
        if input_channel * 2 == output_channel:
            increase_dim = True
            stride = 2
        elif input_channel == output_channel:
            increase_dim = False
            stride = 1
        else:
            raise ValueError('Output and input channel does not match in residual blocks!!!')

        # The first conv layer of the first residual block does not need to be normalized and relu-ed.
        with tf.variable_scope('conv1_in_block'):
            if first_block:
                net = self._conv_layer(input_layer, output_channel)
            else:
                net = self._conv_bn_relu_layer(input_layer, output_channel, stride)


        with tf.variable_scope('conv2_in_block'):
            net = self._conv_bn_relu_layer(net, output_channel, 1)

        # When the channels of input layer and conv2 does not match, we add zero pads to increase the
        #  depth of input layers
        if increase_dim is True:
            pooled_input = self._avg_pool(input_layer)
            padded_input = tf.pad(pooled_input, [[0, 0], [0, 0], [0, 0], [input_channel // 2,
                                                                         input_channel // 2]])
        else:
            padded_input = input_layer

        output = net + padded_input

        return output


    def build(self, input, n):

        with tf.variable_scope('conv0', regularizer=tf.contrib.layers.l2_regularizer(0.0002)):
            net = self._conv_layer(input, 16, tf.nn.relu, name='Conv1')           # 48x48
            net = self._conv_layer(net, 16, tf.nn.relu, name='Conv2')             # 48x48
            net = self._max_pool(net)                               # 24x24
            self.layers.append(net)

        for i in range(n):
            with tf.variable_scope('conv1_%d' % i, regularizer=tf.contrib.layers.l2_regularizer(0.0002)):
                if i == 0:
                    net = self._residual_block(self.layers[-1], 16, first_block=True)    # 24x24
                else:
                    net = self._residual_block(self.layers[-1], 16)
                self.layers.append(net)

        for i in range(n):
            with tf.variable_scope('conv2_%d' % i, regularizer=tf.contrib.layers.l2_regularizer(0.0002)):  # 12x12
                net = self._residual_block(self.layers[-1], 32)
                self.layers.append(net)

        for i in range(n):
            with tf.variable_scope('conv3_%d' % i, regularizer=tf.contrib.layers.l2_regularizer(0.0002)):   # 6x6
                net = self._residual_block(self.layers[-1], 64)
                self.layers.append(net)

        for i in range(n):
            with tf.variable_scope('conv4_%d' % i, regularizer=tf.contrib.layers.l2_regularizer(0.0002)):  # 3x3
                net = self._residual_block(self.layers[-1], 128)
                self.layers.append(net)

            #assert net.get_shape().as_list()[1:] == [3, 3, 128]

        with tf.variable_scope('fc', regularizer=tf.contrib.layers.l2_regularizer(0.0002)):
            net = self._bn_layer(self.layers[-1], self.is_training)
            net = tf.nn.relu(net)
            net = tf.reduce_mean(net, [1, 2])

            assert net.get_shape().as_list()[-1:] == [128]
            net = self._fc_layer(net, self.num_classes)
            self.layers.append(net)

        return self.layers[-1]

    def loss(self, logits, labels):

        labels = tf.cast(labels, tf.int64)
        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits,
                                                                       labels=labels, name='cross_entropy_per_example')
        cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy')
        return cross_entropy_mean

