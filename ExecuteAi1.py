import tensorflow as tf
import numpy as np
import sys

from PreprocessData import loadDataset
from TrainerHelper import splitDataset, getNextBatch, writeValidationResultToFile, getIndexOfOneHot, createNeuralNetwork

arguments = ' '.join(sys.argv[1:])

inputs, labels, labelsDict = loadDataset(arguments, True, False)
n_attributes = len(inputs[1])
n_classes = len(labelsDict)

x = tf.placeholder('float', [None, n_attributes])
y = tf.placeholder('float')

prediction = createNeuralNetwork(x, n_classes, n_attributes)

saver = tf.train.Saver()

with tf.Session() as sess:
    saver.restore(sess, "Ai1Save/Ai1Model.chkpt")

    predictedValue = tf.argmax(prediction, 1)
    feed_dict = {x: inputs, y: labels}
    predictionResult = sess.run(predictedValue, feed_dict=feed_dict)
    writeValidationResultToFile("Ai1_prediction.txt", predictionResult, None, False, labelsDict)
