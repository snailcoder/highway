import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import fully_highway_model

tf.flags.DEFINE_float("learning_rate", 0.001, "Learning rate.")
tf.flags.DEFINE_integer("highway_hidden_size", 50, "Highway block size.")
tf.flags.DEFINE_integer("plain_hidden_size", 71, "Plain layer size.")
tf.flags.DEFINE_integer("num_highway_layer", 9, "Number of highway layer.")
tf.flags.DEFINE_string("data_dir", "/tmp/tensorflow/mnist/input_data",
                       "Directory for storing input data")
tf.flags.DEFINE_integer("batch_size", 100, "Batch size.")
tf.flags.DEFINE_integer("num_epoch", 1000, "Number of epoch.")
tf.flags.DEFINE_integer("eval_step", 5, "Evaluate model after this many steps.")

FLAGS = tf.flags.FLAGS

def main(_):
    mnist = input_data.read_data_sets(FLAGS.data_dir)
    with tf.Graph().as_default():
        config = tf.ConfigProto(log_device_placement=True,
                                allow_soft_placement=True)
        sess = tf.Session(config=config)
        with sess.as_default():
            highway = fully_highway_model.FullyHighwayModel(
                784, FLAGS.highway_hidden_size, FLAGS.plain_hidden_size,
                FLAGS.num_highway_layer, 10, FLAGS.learning_rate)
            X = highway.get_input_X()
            logits = highway.inference(X)
            loss_op = highway.loss(logits)
            global_step = tf.Variable(0, name="global_step", trainable=False)
            train_op = highway.training(loss_op, global_step)
            eval_op = highway.evaluate(logits)
            sess.run(tf.global_variables_initializer())
            num_batch = mnist.train.num_examples / FLAGS.batch_size
            for epoch in range(FLAGS.num_epoch):
                epoch_train_loss = 0.0
                for _ in range(num_batch):
                    batch_X , batch_y = mnist.train.next_batch(FLAGS.batch_size)
                    _, loss, _ = sess.run(
                        [train_op, loss_op, global_step],
                        feed_dict={highway.input_X: batch_X, highway.input_y: batch_y})
                    epoch_train_loss += loss / num_batch
                if epoch % FLAGS.eval_step == 0 or epoch == FLAGS.num_epoch - 1:
                    epoch_eval_accuracy = sess.run(
                        [eval_op],
                        feed_dict={mnist.test.images, mnist.test.labels})
                    print (("Epoch %d, training loss:%f, validation accuracy:%f")
                           % (epoch, epoch_train_loss, epoch_eval_accuracy))

if __name__ == "__main__":
    tf.app.run()