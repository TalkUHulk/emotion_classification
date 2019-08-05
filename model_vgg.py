import tensorflow as tf

class VGG():

  def _max_pool(self, net, name):
    return tf.layers.max_pooling2d(net, pool_size=[2, 2], strides=[2, 2], padding='same', name=name)

  def _conv_layer(self, net, filters, activation=tf.nn.relu, name=None):
    return tf.layers.conv2d(net, filters=filters, kernel_size=[3, 3], strides=[1, 1], padding='same',
                     activation=activation,
                     kernel_initializer=tf.truncated_normal_initializer(stddev=0.1),
                     name=name)


  def _bn_layer(self, net, training, name):
    return tf.layers.batch_normalization(
      net, training=training, name=name)

  def _dropout_layer(self, net, dropout_prob, training, name):
    return tf.layers.dropout(net, rate=dropout_prob, training=training, name=name)

  def _fc_layer(self, net, num_classes, name):
    return tf.layers.dense(net, num_classes, activation=tf.nn.relu, name=name)

  def _conv_fc_layer(self, net, filters, kernel_size, padding='same', activation=None, name=None):
    return tf.layers.conv2d(net, filters=filters, kernel_size=kernel_size, strides=[1, 1], padding=padding,
                     activation=activation,
                     kernel_initializer=tf.truncated_normal_initializer(stddev=0.1),
                     name=name)


  def predict(self, input, num_classes, dropout_prob=0.5, training=True, scope=None):
    with tf.variable_scope(scope, 'VGG', [input]):
      net = self._conv_layer(input, 16, name="conv1_1")         # 48x48x16

      net = self._max_pool(net, 'pool1')            # 24x24x16

      net = self._conv_layer(net, 32, name="conv2_1")  # 24x24x32

      net = self._max_pool(net, 'pool2')  # 12x12x32

      net = self._conv_layer(net, 64, name="conv3_1")  # 12x12x32

      net = self._max_pool(net, 'pool3')  # 6x6x32

      net = self._conv_layer(net, 128, name="conv4_1")  # 6x6x128

      net = self._max_pool(net, 'pool4')  # 3x3x128

      net = self._conv_fc_layer(net, 1024, [3, 3], 'valid', tf.nn.relu, name="fc5")  # 1x1x1024
      net = self._dropout_layer(net, dropout_prob, training, 'dp5')

      net = self._conv_fc_layer(net, 1024, [1, 1], activation=tf.nn.relu, name="fc6")  # 1x1x1024
      net = self._dropout_layer(net, dropout_prob, training, 'dp6')

      net = self._conv_fc_layer(net, num_classes, [1, 1], name="fc7")   # 1x1 x num_classes

      net = tf.squeeze(net, [1, 2], name='fc7/squeezed')

      net = tf.nn.softmax(net, name="prob")

      return net



