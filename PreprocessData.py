import time
import errno
import os
import gc
import numpy as np
import pickle

from sys import getsizeof
from pathlib import Path



# Load all the specified files and fusion their lines only with the specified columns
# Expected parameter format
#
# FileNames: ["msd-jmirderivatives_dev.arff","msd-jmirlpc_dev.arff", "msd-jmirmfccs_dev.arff"]
#
# UsedColumns : [range(5, 90),  for the first supplied file
#                     [1,2,4],  the second
#                     [3,4,5]]  the third...
# return one large data matrix containing all the fusionned lines
#list of files ["msd-jmirderivatives_dev.arff","msd-jmirlpc_dev.arff", "msd-jmirmfccs_dev.arff", "msd-jmirmoments_dev.arff", "msd-jmirspectral_dev.arff", "msd-marsyas_dev_new.arff", "msd-mvd_dev.arff", "msd-rh_dev_new.arff", "msd-ssd_dev.arff", "msd-trh_dev.arff"]

def loadDifferentDataset(OutputFileName, FileNames, UsedColumns):
    DATA_FOLDER_NAME = "Data" + os.sep
     
    # Check if file names are existing
    for filename in FileNames:
        if Path(DATA_FOLDER_NAME + filename).exists() == False:
            raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), filename)
    
    linesInDataFiles = 179555 + 1 # hard coded for simplicity
   
    dataset = [[] for i in range(linesInDataFiles)]
    currentFileIndex = 0

    startTime = time.process_time()

    for filename in FileNames:
        print("Reading ", filename)
        
        with open(DATA_FOLDER_NAME + filename) as file:
            headerSkipped = False
            lineIndex = 0
            for line in file:
                if headerSkipped:
                    lineData = line.split(',')
                    currentDataIndex = 0
                    dataline = dataset[lineIndex]
                    
                    for data in lineData:
                        if currentDataIndex in UsedColumns[currentFileIndex]:
                            dataline.append(float(data))
                            
                        currentDataIndex += 1          

                    lineIndex += 1
                    
                elif "@data" in line:
                    headerSkipped = True
                    
            currentFileIndex += 1
    elapsedTime = time.process_time() - startTime
    print(dataset[linesInDataFiles -2])
    print("Loading dataset Done.")
    print("The process tooked: " + str(elapsedTime) + " seconds")
    print("Cleaning unused memory...")
    gc.collect()

    listMemoryfootprint = getsizeof(dataset)
    lineDatafootprint = getsizeof(dataset[0])
    datasetTotalMemoryFootprint = listMemoryfootprint + lineDatafootprint * len(dataset)
    
    print("dataset estimated memory footprint " + str(datasetTotalMemoryFootprint / (1024 * 1024)) + " MegaBytes")
    input("press to continue")
    return dataset

def getStringFromOneHotVector(oneHotVector, dict):
    for i in range(0,len(oneHotVector)):
        if oneHotVector[i] == 1.0:
            for Label, value in dict.items():
                if value == i:
                    return Label

    return "NO_LABEL_FOUND"

def getStringFromValue(value, dict):
    for Label, value in dict.items():
        if value == i:
            return Label

    return "NO_LABEL_FOUND"

def shuffleDataset(inputs, labels):
    rng_state = np.random.get_state()
    if inputs != None:
        np.random.shuffle(a)
    np.random.set_state(rng_state)

    if labels != None:
        np.random.shuffle(b)

def addStringDataLineToDataset(rawLine, inputMatrix, labelVectorMatrix, dataIndex, labelsDictionnary, attributeQty):
    splittedLine = rawLine.split(',')
  
    offset = len(splittedLine) - attributeQty - 1

    for i in range(2, 2 + attributeQty):
        inputMatrix[dataIndex, i-offset] = float(splittedLine[i])

    labelIndex = labelsDictionnary[splittedLine[len(splittedLine)-1].rstrip()] #dont forget to do rstrip to remove the backslash n at the end of the label 
    labelVectorMatrix[dataIndex, labelIndex] = 1.0 ## initialise in oneHot format [0, 0, 0, 1, 0, 0,]


def loadDataset(fileName, isValidation, pleaseShuffle):
     
    if Path(fileName).exists() == False:
        raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), fileName)
    
    inputQty = 179060 # hard coded for simplicity
    attributeQty = 1   
    outputClassesQty = 0

    try:

        if isValidation == True:
            PreComputedInputs = np.load("ValidationInputs.dat")
            LabelsDict = pickle.load( open( "ValidationLabelsDict.p", "rb" ) )
            print("Precomputed file loaded")
            return PreComputedInputs, None, LabelsDict
        else:
            PreComputedInputs = np.load("TrainInputs.dat")
            PreComputedLabels = np.load("TrainLabels.dat")
            LabelsDict = pickle.load( open( "TrainLabelsDict.p", "rb" ) )
            print("Precomputed files loaded")
            return PreComputedInputs, PreComputedLabels, LabelsDict
       
    except Exception as e:
        print("No precomputed files found continuing normal operations")

    inputs = np.zeros(shape=(1, 1), dtype=np.float32)
    labels = np.zeros(shape=(1, 1), dtype=np.float32)
    labelsStringsDict = {}

    startTime = time.process_time()

    print("Reading ", fileName)
        
    with open(fileName) as file:
        headerSkipped = False
        lineIndex = 0
        isFirstDataLine = True

        for line in file:
            if not headerSkipped and line.startswith("@attribute class"):
                # Creating the label dicionnary by getting the labels inside the brackets {ROCK,PUNK}
                labels = line.split('{')[1].split('}')[0].split(',')
                outputClassesQty = len(labels)
                labelsValues = [x for x in range(0,outputClassesQty)]
                
                for i in range(len(labels)):
                    labelsStringsDict[labels[i]] = labelsValues[i]

                print(labelsStringsDict)
            elif headerSkipped and lineIndex == 0:
                splittedLine = line.split(',')
                attributeQty = len(splittedLine) - 3 #we dont take the two first data cells and the label at the end 
                inputs = np.zeros(shape=(inputQty, attributeQty), dtype=np.float32)
                labels = np.zeros(shape=(inputQty, outputClassesQty), dtype=np.float32)
                
                addStringDataLineToDataset(line, inputs, labels, lineIndex, labelsStringsDict, attributeQty)

                lineIndex += 1
                
            elif headerSkipped:
                addStringDataLineToDataset(line, inputs, labels, lineIndex,labelsStringsDict, attributeQty)

                lineIndex += 1
                    
            elif "@data" in line:
                headerSkipped = True
        print("nunmber of lines read", lineIndex + 1)

    elapsedTime = time.process_time() - startTime
    print("number of input lines:", np.size(inputs,0), "number of attributes:", np.size(inputs,1))
    print("Loading dataset Done.")
    print("The process tooked: " + str(elapsedTime) + " seconds")
    print(inputs[inputQty-2])
    print(inputs[inputQty-1])

    print(getStringFromOneHotVector(labels[inputQty-2],labelsStringsDict))
    print(getStringFromOneHotVector(labels[inputQty-1],labelsStringsDict))
    #print(inputs[np.size(inputs,0)])
    #input("press to continue")

    try:

        if isValidation == True:
            inputs.dump("ValidationInputs.dat")
            pickle.dump( labelsStringsDict, open( "ValidationLabelsDict.p", "wb" ) )
        else:
            inputs.dump("TrainInputs.dat")
            labels.dump("TrainLabels.dat")
            pickle.dump( labelsStringsDict, open( "TrainLabelsDict.p", "wb" ) )
       
    except Exception as e:
        print("Failed to save the datasets to the disk")

    if pleaseShuffle:
        print("Every day im shuffling")
        shuffleDataset(inputs, labels)
    
    print(labels[inputQty-2])
    print(labels[inputQty-1])
    
    return inputs, labels, labelsStringsDict

#loadDataset("resultFile.arff", False, True)