import numpy as np
import math
import time
import errno
import os
import tensorflow as tf

from PreprocessData import getStringFromValue 
from pathlib import Path

def getIndexOfOneHot(oneHotVector):
	for i in range(0, len(oneHotVector)):
		if oneHotVector[i] == 1.0:
			return i

def writeValidationResultToFile(fileName, predictedValue, actualValue, hasExpectedValue, labelsDict):

	with open(fileName,'w') as file:
		if hasExpectedValue:

			for i in range(0, len(predictedValue)):
				predicted = getStringFromValue(predictedValue[i], labelsDict)
				expected = getStringFromValue(getIndexOfOneHot(actualValue[i]), labelsDict)
				line = predicted + ", " + expected + "\n"
				file.write(line)
		else:
			for i in range(0, len(predictedValue)):
				predicted = getStringFromValue(predictedValue[i], labelsDict) + "\n"
				file.write(predicted)


def getNextBatch(inputs, labels, batchSize, currentBatchIndex):
	assert(len(inputs) == len(labels))
	
	thisBatchStartIndex = currentBatchIndex *  batchSize
	thisBatchStartEndIndex = thisBatchStartIndex + batchSize

	thisBatchInputs = inputs[thisBatchStartIndex:thisBatchStartEndIndex]
	thisBatchLabels = labels[thisBatchStartIndex:thisBatchStartEndIndex]

	return thisBatchInputs, thisBatchLabels

def splitDataset(inputs, labels, trainingPercentage):
	
	assert(len(inputs) == len(labels))
	assert(trainingPercentage <= 1.0 and trainingPercentage >= 0)

	print("Total elements", len(inputs))

	n_Train = math.ceil(len(inputs) * trainingPercentage)
	n_Valid = math.floor(len(inputs) * (1.0 - trainingPercentage))

	print("train qty", n_Train, "Validation", n_Valid)
	print("total", n_Train + n_Valid)

	trainInputs = inputs[0:n_Train]
	trainLabels = labels[0:n_Train]

	validationInputs = inputs[n_Train:len(inputs)]
	validationLabels = labels[n_Train:len(inputs)]

	print("trainInputs",  len(trainInputs), "trainLabels",  len(trainLabels))
	print("validationInputs",  len(validationInputs), "validationLabels",  len(validationLabels))
	print("Total inputs", len(trainInputs) + len(validationInputs), "Total labels", len(trainLabels)+len(validationLabels))

	return trainInputs, trainLabels, validationInputs, validationLabels

def neural_network_model(data, nbClass, nbattributes):
	n_nodes_hl1 = 500
	n_nodes_hl2 = 500
	n_nodes_hl3 = 500
	n_nodes_hl4 = 500
	n_nodes_hl5 = 500
	n_nodes_hl6 = 500
	n_nodes_hl7 = 500
	n_nodes_hl8 = 500
	n_nodes_hl9 = 500
	n_nodes_hl10 = 500

	hidden_1_layer = {'weights':tf.Variable(tf.random_normal([nbattributes, n_nodes_hl1])), 'biases':tf.Variable(tf.random_normal([n_nodes_hl1]))}
	hidden_2_layer = {'weights':tf.Variable(tf.random_normal([n_nodes_hl1, n_nodes_hl2])), 'biases':tf.Variable(tf.random_normal([n_nodes_hl2]))}
	hidden_3_layer = {'weights':tf.Variable(tf.random_normal([n_nodes_hl2, n_nodes_hl3])), 'biases':tf.Variable(tf.random_normal([n_nodes_hl3]))}
	hidden_4_layer = {'weights':tf.Variable(tf.random_normal([n_nodes_hl3, n_nodes_hl4])), 'biases':tf.Variable(tf.random_normal([n_nodes_hl4]))}
	hidden_5_layer = {'weights':tf.Variable(tf.random_normal([n_nodes_hl4, n_nodes_hl5])), 'biases':tf.Variable(tf.random_normal([n_nodes_hl5]))}
	hidden_6_layer = {'weights':tf.Variable(tf.random_normal([n_nodes_hl5, n_nodes_hl6])), 'biases':tf.Variable(tf.random_normal([n_nodes_hl6]))}
	hidden_7_layer = {'weights':tf.Variable(tf.random_normal([n_nodes_hl6, n_nodes_hl7])), 'biases':tf.Variable(tf.random_normal([n_nodes_hl7]))}
	hidden_8_layer = {'weights':tf.Variable(tf.random_normal([n_nodes_hl7, n_nodes_hl8])), 'biases':tf.Variable(tf.random_normal([n_nodes_hl8]))}
	hidden_9_layer = {'weights':tf.Variable(tf.random_normal([n_nodes_hl8, n_nodes_hl9])), 'biases':tf.Variable(tf.random_normal([n_nodes_hl9]))}
	hidden_10_layer = {'weights':tf.Variable(tf.random_normal([n_nodes_hl9, n_nodes_hl10])), 'biases':tf.Variable(tf.random_normal([n_nodes_hl10]))}
	output_layer = {'weights':tf.Variable(tf.random_normal([n_nodes_hl10, nbClass])), 'biases':tf.Variable(tf.random_normal([nbClass])),}

	l1 = tf.add(tf.matmul(data,hidden_1_layer['weights']), hidden_1_layer['biases'])
	l1 = tf.nn.relu(l1)

	l2 = tf.add(tf.matmul(l1,hidden_2_layer['weights']), hidden_2_layer['biases'])
	l2 = tf.nn.relu(l2)

	l3 = tf.add(tf.matmul(l2,hidden_3_layer['weights']), hidden_3_layer['biases'])
	l3 = tf.nn.relu(l3)

	l4 = tf.add(tf.matmul(l3,hidden_4_layer['weights']), hidden_4_layer['biases'])
	l4 = tf.nn.relu(l4)

	l5 = tf.add(tf.matmul(l4,hidden_5_layer['weights']), hidden_5_layer['biases'])
	l5 = tf.nn.relu(l5)

	l6 = tf.add(tf.matmul(l5,hidden_6_layer['weights']), hidden_6_layer['biases'])
	l6 = tf.nn.relu(l6)

	l7 = tf.add(tf.matmul(l6,hidden_7_layer['weights']), hidden_7_layer['biases'])
	l7 = tf.nn.relu(l7)

	l8 = tf.add(tf.matmul(l7,hidden_8_layer['weights']), hidden_8_layer['biases'])
	l8 = tf.nn.relu(l8)

	l9 = tf.add(tf.matmul(l8,hidden_9_layer['weights']), hidden_9_layer['biases'])
	l9 = tf.nn.relu(l9)

	l10 = tf.add(tf.matmul(l9,hidden_10_layer['weights']), hidden_10_layer['biases'])
	l10 = tf.nn.relu(l10)

	output = tf.matmul(l10,output_layer['weights']) + output_layer['biases']

	return output
