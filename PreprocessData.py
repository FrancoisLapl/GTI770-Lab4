import time
import errno
import os
import gc
import numpy as np

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

def addStringDataLineToDataset(rawLine, inputMatrix, labelVectorMatrix, dataIndex, labelsDictionnary, attributeQty):
    splittedLine = rawLine.split(',')
  
    offset = len(splittedLine) - attributeQty - 1

    for i in range(2, 2 + attributeQty):
        inputMatrix[dataIndex, i-offset] = float(splittedLine[i])

    labelIndex = labelsDictionnary[splittedLine[len(splittedLine)-1].rstrip()] #dont forget to do rstrip to remove the backslash n at the end of the label 
    labelVectorMatrix[dataIndex, labelIndex] = 1.0 ## initialise in oneHot format [0, 0, 0, 1, 0, 0,]


def loadDataset(fileName, PleaseLoadLabels):
     
    if Path(fileName).exists() == False:
        raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), fileName)
    
    inputQty = 179555 + 1 # hard coded for simplicity
    attributeQty = 1360   
    outputClassesQty = 0

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
                # Creating the label dicionnary
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

    elapsedTime = time.process_time() - startTime
    print("number of input lines:", np.size(inputs,0), "number of attributes:", np.size(inputs,1))
    print("Loading dataset Done.")
    print("The process tooked: " + str(elapsedTime) + " seconds")
    print(inputs[np.size(inputs,0)]-1)
    #print(inputs[np.size(inputs,0)])
    input("press to continue")

    return inputs, labels

loadDataset("resultFile.arff", True)