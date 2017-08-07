import numpy as np
import math

def getNextBatch(inputs, labels, batchSize, currentBatch):
	return

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

	validationInputs = inputs[n_Train+1:len(inputs)]
	validationLabels = labels[n_Train+1:len(inputs)]
	
	print("trainInputs",  len(trainInputs), "trainLabels",  len(trainLabels))
	print("validationInputs",  len(validationInputs), "validationLabels",  len(validationLabels))
	print("Total inputs", len(trainInputs) + len(validationInputs), "Total labels", len(trainLabels)+len(validationLabels))

	return trainInputs, trainLabels, validationInputs, validationLabels
