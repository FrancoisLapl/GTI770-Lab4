import tensorflow as tf
import numpy as np
import sys

from PreprocessData import loadDataset
from TrainerHelper import splitDataset, getNextBatch, writeValidationResultToFile, getIndexOfOneHot, createNeuralNetwork

arguments = ' '.join(sys.argv[1:])

inputs, labels, labelsDict = loadDataset(arguments, False, True)

n_attributes = len(inputs[1])
n_classes = len(labels[0])

train_Inputs, train_Labels, validation_Inputs, validation_Labels = splitDataset(inputs,labels, 0.05) 

batch_size = 100

x = tf.placeholder('float', [None, n_attributes])
y = tf.placeholder('float')

def train_neural_network(x):
    prediction = createNeuralNetwork(x, n_classes, n_attributes)
    cost = tf.reduce_mean( tf.nn.softmax_cross_entropy_with_logits(logits=prediction,labels=y) )
    optimizer = tf.train.AdamOptimizer().minimize(cost)
    saver = tf.train.Saver() 
    
    n_epochs = 2

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        for epoch in range(n_epochs):
            epoch_loss = 0
            for i in range(int(len(train_Inputs)/batch_size)):
                epoch_x, epoch_y = getNextBatch(train_Inputs, train_Labels, batch_size, i)
                i, c = sess.run([optimizer, cost], feed_dict={x: epoch_x, y: epoch_y})
                epoch_loss += c

            print('Epoch', epoch, 'completed out of', n_epochs,'loss:', epoch_loss)
        saver.save(sess, 'Ai1Save/Ai1Model.chkpt')

        predictedValue = tf.argmax(prediction, 1)
        #feed_dict = {x: validation_Inputs, y: validation_Labels}

        #predictionResult = sess.run(predictedValue, feed_dict=feed_dict)
        #writeValidationResultToFile("Ai1_prediction.txt", predictionResult, validation_Labels, True, labelsDict)

        correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
        #print(correct)
        accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
        print('Accuracy:', accuracy.eval({x:validation_Inputs, y:validation_Labels}))

train_neural_network(x)
