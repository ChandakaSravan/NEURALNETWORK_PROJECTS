# Chandaka, Sravan
# 1002_059_166
# 2023_04_17
# Assignment_04_02

import tensorflow as tf
import numpy as np
import os
import keras
import pytest
import tensorflow.keras as keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras import utils as utils
from tensorflow.keras.datasets import mnist
from Chandaka_04_01 import CNN

  

def test_train_data():
    X_train,y_train,X_test,y_test=get_data()
    
    batch_size=20
    num_epochs=15
    matrix=utils.to_categorical(y_test)
    num_of_classes=matrix.shape[1]
    model=CNN()
    model.add_input_layer(shape=(28,28,1),name="inputlayers")
    model.append_conv2d_layer(16,(3,3),activation="relu",name="conlayer01")
    model.append_maxpooling2d_layer(pool_size=(3,3),padding="same",strides=2,name="poollayer01")
    model.append_conv2d_layer(64,(3,3),activation="relu",name="conlayer02")
    model.append_flatten_layer(name="flattenlayer")
    model.append_dense_layer(num_nodes=32,activation="sigmoid",name="denselayer01")
    model.append_dense_layer(num_nodes=10,activation="softmax",name="denselayer02")
    model.set_loss_function(loss="SparseCategoricalCrossentropy")
    model.set_optimizer(optimizer="SGD")
    model.set_metric("Accuracy")
    loss=model.train(X_train,y_train,batch_size=batch_size,num_epochs=num_epochs)
    assert loss[-1] < loss[0]
   # print(loss)


def get_data():
    # Load the MNIST dataset
    (X_train_full, y_train_full), (X_test_full, y_test_full) = tf.keras.datasets.mnist.load_data()

    # Extract a subset of the data
    num_samples = 500
    X_train = X_train_full[:num_samples]
    y_train = y_train_full[:num_samples]
    X_test = X_test_full[:num_samples]
    y_test = y_test_full[:num_samples]

    # Reshape the data and scale pixel values to [0, 1]
    X_train = np.expand_dims(X_train, axis=-1).astype('float32') / 255
    X_test = np.expand_dims(X_test, axis=-1).astype('float32') / 255

    return X_train, y_train, X_test, y_test

#Evaluation function
def test_evalute():
    (X_train_full, y_train_full), (X_test_full, y_test_full) = keras.datasets.mnist.load_data()
    no_samp = 500
    y_train = y_train_full[:no_samp]
    X_train = X_train_full[:no_samp]
    y_test = y_test_full[:no_samp]
    X_test = X_test_full[:no_samp]

    X_test = X_test[..., None] / 255.0
    X_train = X_train[..., None] / 255.0

    model = CNN()
    model.add_input_layer(shape=X_train.shape[1:], name="inputlayers")
    model.append_conv2d_layer(16, (3,3), activation="relu", name="conlayer01")
    model.append_maxpooling2d_layer(pool_size=(2,2), padding="same", strides=2, name="poollayer01")
    model.append_conv2d_layer(64, (3,3), activation="relu", name="conlayer02")
    model.append_flatten_layer(name="flattenlayer")
    model.append_dense_layer(num_nodes=32, activation="relu", name="denselayer01")
    model.append_dense_layer(num_nodes=10, activation="softmax", name="denselayer02")
    model.set_loss_function(loss="SparseCategoricalCrossentropy")
    model.set_optimizer(optimizer="SGD")
    model.set_metric("Accuracy")

    batch_size = 25
    num_epochs = 17
    history = model.train(X_train, y_train, batch_size, num_epochs=num_epochs)
    loss, accuracy = model.evaluate(X_test, y_test)

   # print(f"Loss: {loss}")
   # print(f"Accuracy: {accuracy}")
    assert accuracy > 0.71
    assert loss < 2.3

  
#test_train_data()
#test_evalute()
