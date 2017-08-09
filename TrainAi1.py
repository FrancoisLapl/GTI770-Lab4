import tensorflow as tf
import numpy as np
import sys

from PreprocessData import loadDataset
from TrainerHelper import splitDataset, getNextBatch, writeValidationResultToFile, getIndexOfOneHot, createNeuralNetwork

arguments = ' '.join(sys.argv[1:])

inputs, labels, labelsDict = loadDataset(arguments, False, True)

n_attributes = len(inputs[1])
n_classes = len(labels[0])

train_Inputs, train_Labels, validation_Inputs, validation_Labels = splitDataset(inputs,labels, 0.7) 

batch_Qty = 100

dataPlaceHolder = tf.placeholder('float', [None, n_attributes])
y = tf.placeholder('float')

def train_neural_network(x):
    nn = createNeuralNetwork(x, n_classes, n_attributes)
    cost = tf.reduce_mean( tf.nn.softmax_cross_entropy_with_logits(logits=nn,labels=y) )
    opti = tf.train.AdamOptimizer().minimize(cost)
    saver = tf.train.Saver() 
    
    n_epochs = 20

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        for epoch in range(n_epochs):
            for i in range(int(len(train_Inputs)/batch_Qty)):
                epoch_x, epoch_y = getNextBatch(train_Inputs, train_Labels, batch_Qty, i)
                i, c = sess.run([opti, cost], feed_dict={x: epoch_x, y: epoch_y})

            print('Epoch', epoch, 'completed out of', n_epochs)
        saver.save(sess, 'Ai1Save/Ai1Model.chkpt')

train_neural_network(dataPlaceHolder)
