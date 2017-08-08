import tensorflow as tf
import numpy as np
import sys

from PreprocessData import loadDataset
from TrainerHelper import splitDataset, getNextBatch, writeValidationResultToFile, getIndexOfOneHot

arguments = ' '.join(sys.argv[1:])

inputs, labels, labelsDict = loadDataset(arguments, False, False)

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

n_attributes = len(inputs[1])
n_classes = len(labels[0])

train_Inputs, train_Labels, validation_Inputs, validation_Labels = splitDataset(inputs,labels, 0.05) 

batch_size = 100

x = tf.placeholder('float', [None, n_attributes])
y = tf.placeholder('float')

def neural_network_model(data):
    hidden_1_layer = {'weights':tf.Variable(tf.random_normal([n_attributes, n_nodes_hl1])),
                      'biases':tf.Variable(tf.random_normal([n_nodes_hl1]))}

    hidden_2_layer = {'weights':tf.Variable(tf.random_normal([n_nodes_hl1, n_nodes_hl2])),
                      'biases':tf.Variable(tf.random_normal([n_nodes_hl2]))}

    hidden_3_layer = {'weights':tf.Variable(tf.random_normal([n_nodes_hl2, n_nodes_hl3])),
                      'biases':tf.Variable(tf.random_normal([n_nodes_hl3]))}

    hidden_4_layer = {'weights':tf.Variable(tf.random_normal([n_nodes_hl3, n_nodes_hl4])),
                      'biases':tf.Variable(tf.random_normal([n_nodes_hl4]))}

    hidden_5_layer = {'weights':tf.Variable(tf.random_normal([n_nodes_hl4, n_nodes_hl5])),
                      'biases':tf.Variable(tf.random_normal([n_nodes_hl5]))}

    hidden_6_layer = {'weights':tf.Variable(tf.random_normal([n_nodes_hl5, n_nodes_hl6])),
                      'biases':tf.Variable(tf.random_normal([n_nodes_hl6]))}

    hidden_7_layer = {'weights':tf.Variable(tf.random_normal([n_nodes_hl6, n_nodes_hl7])),
                      'biases':tf.Variable(tf.random_normal([n_nodes_hl7]))}

    hidden_8_layer = {'weights':tf.Variable(tf.random_normal([n_nodes_hl7, n_nodes_hl8])),
                      'biases':tf.Variable(tf.random_normal([n_nodes_hl8]))}

    hidden_9_layer = {'weights':tf.Variable(tf.random_normal([n_nodes_hl8, n_nodes_hl9])),
                      'biases':tf.Variable(tf.random_normal([n_nodes_hl9]))}

    hidden_10_layer = {'weights':tf.Variable(tf.random_normal([n_nodes_hl9, n_nodes_hl10])),
                      'biases':tf.Variable(tf.random_normal([n_nodes_hl10]))}

    output_layer = {'weights':tf.Variable(tf.random_normal([n_nodes_hl10, n_classes])),
                    'biases':tf.Variable(tf.random_normal([n_classes])),}


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

def train_neural_network(x):
    prediction = neural_network_model(x)
    cost = tf.reduce_mean( tf.nn.softmax_cross_entropy_with_logits(logits=prediction,labels=y) )
    optimizer = tf.train.AdamOptimizer().minimize(cost)

    hm_epochs = 2
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        for epoch in range(hm_epochs):
            epoch_loss = 0
            for i in range(int(len(train_Inputs)/batch_size)):
                epoch_x, epoch_y = getNextBatch(train_Inputs, train_Labels, batch_size, i)
                #print(i)
                #print(epoch_x)
                #print(epoch_y)
                i, c = sess.run([optimizer, cost], feed_dict={x: epoch_x, y: epoch_y})
                epoch_loss += c

            print('Epoch', epoch, 'completed out of',hm_epochs,'loss:', epoch_loss)
        #saver = tf.train.Saver() 
        #saver.save(sess, 'Ai1_Model')
        predictedValue = tf.argmax(prediction, 1)
        feed_dict = {x: validation_Inputs, y: validation_Labels}

        predictionResult = sess.run(predictedValue, feed_dict=feed_dict)
        writeValidationResultToFile("Ai1_prediction.txt", predictionResult, validation_Labels, True, labelsDict)

        ##correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
        ##print(correct)
        ##accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
        ##print('Accuracy:', accuracy.eval({x:validation_Inputs, y:validation_Labels}))

train_neural_network(x)
