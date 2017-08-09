# GTI770-Lab4

Pre-requisite
-A computer with at least 6 Gb of ram
-TensorFlow 1.21
-python 3.5 64 bits (IMPORTANT for big datasets)
-numpy

Step to set up and run the project

1. in the GTI770_FileCombinator folder copy to the root of the repository the two files 
	-GTI770_FileCombinator.jar
	-GTI770_FileCombinator.iml
2. Download and extract the two specified archive at the root of the repository
	-DEVNEW.zip
	-NOLABELSNEW.zip

3. Now your repository root directory should look like this 
	-GTI770_FileCombinator.jar
	-GTI770_FileCombinator.iml
	-DEVNEW (with all the labeled .arff files)
	-NOLABELSNEW (with all the unlabeled .arff files)
	-all the python scripts
	-and maybe some other unecessary stuff

4. Now you must create the actual files needed for the training/validation and test of our algorythm (this took 10 minutes on an SSD)
	- please execute the jar program to merge/clean/remove the datasets with this command: java -jar GTI770_FileCombinator.jar 

5.a To train the algorythms please run (this may be long depending of you hardware/tensorFlow set-up)
	- Python TrainAi1.py train.arff
	- Python TrainAi2.py train.arff (NOT IMPLEMENTED)

5.b alternatively to evaluate the algorythms please run this will produce two file containing the result of the prediction of the two Ai's (AI1_output.txt and AI2_output.txt)
	- Python ExecuteAi1.py valid.arff
	- Python ExecuteAi2.py valid.arff (NOT IMPLEMENTED)