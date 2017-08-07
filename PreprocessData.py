import time
import errno
import os
import gc

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

def loadDataset(OutputFileName, FileNames, UsedColumns, usePrebakedModule = False):
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
    
loadDataset("Dataset1",["msd-jmirderivatives_dev.arff","msd-jmirlpc_dev.arff", "msd-jmirmfccs_dev.arff", "msd-jmirmoments_dev.arff", "msd-jmirspectral_dev.arff", "msd-marsyas_dev_new.arff", "msd-mvd_dev.arff", "msd-rh_dev_new.arff", "msd-ssd_dev.arff", "msd-trh_dev.arff"]
             ,[range(5, 90),                range(5,15),            range(5,20),               range(5,11),                range(2,17),                 range(2,126),                range(2,420),        range(2,60),           range(2,168),        range(2,420)])
