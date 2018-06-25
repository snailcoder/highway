import tensorflow as tf

class FullyHighwayModel(object):
    def __init__(self, input_size, highway_hidden_size,
                 num_highway_layer, num_class, learning_rate):
        self.highway_hidden_size = highway_hidden_size
        self.num_highway_layer = num_highway_layer
        self.num_class = num_class
        self.learning_rate = learning_rate
        self.input_X = tf.placeholder(
            tf.float32, [None, input_size], name="input_X")
        self.input_y = tf.placeholder(
            tf.float32, [None, num_class], name="input_y")

    def _highway_layer(self, X, hidden_size, H_activation, T_activation):
        input_size = X.shape[1].value
        W_H = tf.Variable(
            tf.truncated_normal(
                [input_size, hidden_size]), name="W_H")
        b_H = tf.Variable(tf.zeros([hidden_size], tf.float32), name="b_H")
        H_out = H_activation(tf.nn.xw_plus_b(X, W_H, b_H, name="H_out"))
        W_T = tf.Variable(
            tf.truncated_normal(
                [input_size, hidden_size]), name="W_T")
        b_T = tf.Variable(tf.zeros([hidden_size], tf.float32), name="b_T")
        T_out = T_activation(tf.nn.xw_plus_b(X, W_T, b_T, name="T_out"))
        C_out = 1 - T_out
        output = tf.multiply(H_out, T_out) + tf.multiply(X, C_out)
        return output

    def _plain_layer(self, X, hidden_size, activation):
        input_size = X.shape[1].value
        W = tf.Variable(
            tf.truncated_normal([input_size, hidden_size]))
        b = tf.Variable(tf.zeros([hidden_size], tf.float32))
        return activation(tf.matmul(X, W) + b)

    def get_input_X(self):
        return self.input_X

    def get_input_y(self):
        return self.input_y

    def training(self, loss, global_step, batch_size,
                 decay_rate, decay_steps, decay_epochs):
        with tf.name_scope("trainning"):
            learning_rate = tf.train.exponential_decay(
                self.learning_rate, global_step * batch_size // decay_epochs,
                decay_steps, decay_rate, staircase=True)
            # Using SGD with momentum.
            optimizer = tf.train.MomentumOptimizer(learning_rate, 0.9)
            # optimizer = tf.train.GradientDescentOptimizer(
            #     learning_rate=self.learning_rate)
            train_op = optimizer.minimize(loss, global_step=global_step)
            return train_op

    def inference(self, X):
        logits = self._plain_layer(X, self.highway_hidden_size, tf.sigmoid)
        for _ in range(self.num_highway_layer):
            logits = self._highway_layer(logits, self.highway_hidden_size,
                                         tf.sigmoid, tf.sigmoid)
        logits = self._plain_layer(logits, self.num_class, tf.sigmoid)
        return logits

    def loss(self, logits):
        with tf.name_scope("loss"):
            xentropy = tf.nn.softmax_cross_entropy_with_logits(
                labels=self.input_y, logits=logits, name="cross_entropy")
            mean_loss = tf.reduce_mean(xentropy, axis=0)
            return mean_loss

    def evaluate(self, logits):
        with tf.name_scope("eval"):
            pred = tf.argmax(logits, axis=1, name="pred")
            correct = tf.argmax(self.input_y, axis=1, name="correct")
            correct_pred = tf.equal(pred, correct)
            accuracy = tf.reduce_mean(
                tf.cast(correct_pred, "float"), name="accuracy")
            return accuracy