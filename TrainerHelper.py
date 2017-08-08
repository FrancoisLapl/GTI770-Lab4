import numpy as np
import math
import time
import errno
import os
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