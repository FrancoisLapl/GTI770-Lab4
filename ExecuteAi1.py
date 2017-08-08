import tensorflow as tf
import numpy as np
import sys

from PreprocessData import loadDataset


arguments = ' '.join(sys.argv[1:])

inputs, labels, labelsDict = loadDataset(arguments, True, False)


saver = tf.train.Saver()
