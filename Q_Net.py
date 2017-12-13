import tensorflow as tf

class Q_Net:

    def __init__(self, input_size=None, output_size=None):

        self.inputs = tf.placeholder(shape=[1, input_size], dtype=tf.float32)
        self.W = tf.Variable(tf.random_uniform([input_size, output_size], 0, 0.01))
        self.Q = tf.matmul(self.inputs, self.W)
        self.predict = tf.argmax(self.Q, 1)

        self.next_Q = tf.placeholder(shape=[1,output_size], dtype=tf.float32)
        self.loss = tf.reduce_sum(tf.square(self.next_Q - self.Q))
        self.optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.1)
        self.update = self.optimizer.minimize(self.loss)


    def initialize(self):
        tf.initialize_all_variables()






