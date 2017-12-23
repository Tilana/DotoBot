import tensorflow as tf

class Q_Net:

    def __init__(self, input_size=None, output_size=None):


        self.inputs = tf.placeholder(shape=[1, input_size, 1], dtype=tf.float32)

        self.conv = tf.layers.conv1d(inputs=self.inputs, filters=4, kernel_size=(5), padding='same', activation=tf.nn.relu)
        self.pool1 = tf.layers.max_pooling1d(inputs=self.conv, pool_size=(2), strides=1)

        self.conv2 = tf.layers.conv1d(inputs=self.pool1, filters=4, kernel_size=[5], padding='same', activation=tf.nn.relu)
        self.pool2 = tf.layers.max_pooling1d(inputs=self.conv2, pool_size=[2], strides=1)

        self.pool2_flat = tf.reshape(self.pool2, [-1, 20*4])
        self.dense = tf.layers.dense(inputs=self.pool2_flat, units=30, activation=tf.nn.relu)

        self.logits = tf.layers.dense(inputs=self.dense, units=20)

        self.predict = tf.argmax(self.logits, axis=1)

        self.target_Q = tf.placeholder(shape=[1,output_size], dtype=tf.float32)
        self.loss = tf.reduce_sum(tf.square(self.target_Q - self.logits))
        self.optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.005)
        self.update = self.optimizer.minimize(self.loss)


    def initialize(self):
        tf.initialize_all_variables()






