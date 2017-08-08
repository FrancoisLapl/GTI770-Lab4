import tensorflow as tf
import numpy as np

with tf.Session() as sess:
	new_saver = tf.train.import_meta_graph('Ai1_Model.meta')
	new_saver.restore(sess, tf.train.latest_checkpoint('./'))