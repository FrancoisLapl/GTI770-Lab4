package com.fsmimeault;

import java.io.*;
import java.util.ArrayList;
import java.util.List;

public class Main {

    public static void main(String[] args) throws IOException {

        // Merge training files
        System.out.println("Starting the merge of training files.....................................");
        List<BufferedReader> bufferedReaderListTrain = new ArrayList<>();
        LoadFiles(bufferedReaderListTrain, true);

        FileOutputStream trainFileOutputStream = new FileOutputStream("train.arff");
        BufferedWriter trainBufferedWriter = new BufferedWriter(new OutputStreamWriter(trainFileOutputStream));

        trainBufferedWriter.write("@relation Combined File of training music track primitives");
        trainBufferedWriter.newLine();
        trainBufferedWriter.newLine();

        WriteAttributesToFile(bufferedReaderListTrain, trainBufferedWriter);
        WriteDataToFile(bufferedReaderListTrain, trainBufferedWriter);
        System.out.println("Finished the merge of training files.....................................");

        // Close all buffers
        System.out.println("Closing training buffers........................................");
        CloseBuffers(bufferedReaderListTrain);
        trainBufferedWriter.close();

        // Merge validation files
        System.out.println("Starting the merge of training files.....................................");
        List<BufferedReader> bufferedReaderListValid = new ArrayList<>();
        LoadFiles(bufferedReaderListValid, false);

        FileOutputStream validFileOutputStream = new FileOutputStream("valid.arff");
        BufferedWriter validBufferedWriter = new BufferedWriter(new OutputStreamWriter(validFileOutputStream));

        validBufferedWriter.write("@relation Combined File of validation music track primitives");
        validBufferedWriter.newLine();
        validBufferedWriter.newLine();

        WriteAttributesToFile(bufferedReaderListValid, validBufferedWriter);
        WriteDataToFile(bufferedReaderListValid, validBufferedWriter);
        System.out.println("Finished the merge of validation files.....................................");

        // Close all buffers
        System.out.println("Closing validation buffers........................................");
        CloseBuffers(bufferedReaderListValid);
        validBufferedWriter.close();
    }

    private static void LoadFiles(List<BufferedReader> bufferedReaderList, boolean train) throws FileNotFoundException {

        if (train){
            bufferedReaderList.add(OpenFile("DEVNEW/msd-jmirderivatives_dev.arff"));
            bufferedReaderList.add(OpenFile("DEVNEW/msd-jmirlpc_dev.arff"));
            bufferedReaderList.add(OpenFile("DEVNEW/msd-jmirmfccs_dev.arff"));
            bufferedReaderList.add(OpenFile("DEVNEW/msd-jmirmoments_dev.arff"));
            bufferedReaderList.add(OpenFile("DEVNEW/msd-jmirspectral_dev.arff"));
            bufferedReaderList.add(OpenFile("DEVNEW/msd-marsyas_dev_new.arff"));
            bufferedReaderList.add(OpenFile("DEVNEW/msd-mvd_dev.arff"));
            bufferedReaderList.add(OpenFile("DEVNEW/msd-rh_dev_new.arff"));
            bufferedReaderList.add(OpenFile("DEVNEW/msd-ssd_dev.arff"));
            bufferedReaderList.add(OpenFile("DEVNEW/msd-trh_dev.arff"));
        } else {
            bufferedReaderList.add(OpenFile("NOLABELSNEW/msd-jmirderivatives_test_nolabels.arff"));
            bufferedReaderList.add(OpenFile("NOLABELSNEW/msd-jmirlpc_test_nolabels.arff"));
            bufferedReaderList.add(OpenFile("NOLABELSNEW/msd-jmirmfccs_test_nolabels.arff"));
            bufferedReaderList.add(OpenFile("NOLABELSNEW/msd-jmirmoments_test_nolabels.arff"));
            bufferedReaderList.add(OpenFile("NOLABELSNEW/msd-jmirspectral_test_nolabels.arff"));
            bufferedReaderList.add(OpenFile("NOLABELSNEW/msd-marsyas_test_new_nolabels.arff"));
            bufferedReaderList.add(OpenFile("NOLABELSNEW/msd-mvd_test_nolabels.arff"));
            bufferedReaderList.add(OpenFile("NOLABELSNEW/msd-rh_test_nolabels.arff"));
            bufferedReaderList.add(OpenFile("NOLABELSNEW/msd-ssd_test_nolabels.arff"));
            bufferedReaderList.add(OpenFile("NOLABELSNEW/msd-trh_test_nolabels.arff"));
        }
    }

    private static void WriteDataToFile(List<BufferedReader> bufferedReaderList, BufferedWriter bufferedWriter) throws IOException {

        SkipLineAndPrint(bufferedReaderList, bufferedWriter, "@data");

        while (bufferedReaderList.iterator().next().ready()){

            System.out.println("Hey, we're still working here! Processing the next item...");

            boolean firstFile = true;
            String sampleID = "";
            String trackID = "";
            String label = "";
            String input;
            String lineToPrint = "";

            boolean brokenSampleToDrop = false;

            for (BufferedReader bufferedReader :
                    bufferedReaderList) {

                input = bufferedReader.readLine();
                String[] result = input.split(",");

                if (firstFile){
                    sampleID = result[0];
                    trackID = result[1];
                    label = result[result.length - 1];

                    lineToPrint = sampleID + "," + trackID;
                    firstFile = false;
                } else {
                    // make sure the data is not fucked up
                    if (!sampleID.contentEquals(result[0])){
                        System.out.println("Yio mate sample ID is fucked up");
                        brokenSampleToDrop = true;
                    }

                    if (!trackID.contentEquals(result[1])){
                        System.out.println("Yio mate track ID is fucked up");
                        brokenSampleToDrop = true;
                    }

                    if (!label.contentEquals(result[result.length - 1])){
                        System.out.println("Yio mate label is fucked up on sampleID : " + sampleID + ". The original label was " + label + " and the current label is " + result[result.length - 1]);
                        brokenSampleToDrop = true;
                    }
                }

                for (int i = 2; i < result.length - 1; i++){
                    lineToPrint += "," + result[i];
                }
            }

            lineToPrint += "," + label;

            if (!brokenSampleToDrop){
                bufferedWriter.write(lineToPrint);
                bufferedWriter.newLine();
            }
        }


    }

    private static void SkipLineAndPrint(List<BufferedReader> bufferedReaderList, BufferedWriter bufferedWriter, String s) throws IOException {
        for (BufferedReader bufferedReader :
                bufferedReaderList) {
            bufferedReader.readLine();
        }

        bufferedWriter.write(s);
        bufferedWriter.newLine();
    }

    private static void WriteAttributesToFile(List<BufferedReader> bufferedReaderList, BufferedWriter bufferedWriter) throws IOException {
        String input;
        String lastAttribute = "";
        boolean firstFile = true;

        for (BufferedReader bufferedReader :
                bufferedReaderList) {

            bufferedReader.readLine();
            bufferedReader.readLine();

            // Print the SAMPLEID AND TRACKID only on first file
            if (firstFile){
                firstFile = false;
                bufferedWriter.write(bufferedReader.readLine());
                bufferedWriter.newLine();

                bufferedWriter.write(bufferedReader.readLine());
                bufferedWriter.newLine();
            } else {
                // If not first file just skip the first two attributes
                bufferedReader.readLine();
                bufferedReader.readLine();
            }

            input = bufferedReader.readLine();

            while (input.startsWith("@attribute")){
                if (input.startsWith("@attribute class")){
                    lastAttribute = input;
                } else {
                    bufferedWriter.write(input);
                    bufferedWriter.newLine();
                }

                input = bufferedReader.readLine();
            }
        }

        bufferedWriter.write(lastAttribute);
        bufferedWriter.newLine();
        bufferedWriter.newLine();
    }

    private static void CloseBuffers(List<BufferedReader> bufferedReaderList) throws IOException {
        for (BufferedReader bufferedReader :
                bufferedReaderList) {
            bufferedReader.close();
        }
    }

    private static BufferedReader OpenFile(String filePath) throws FileNotFoundException {
        return new BufferedReader(new InputStreamReader(new FileInputStream(filePath)));
    }
}
