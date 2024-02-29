# Chandaka, Sravan
# 1002_059_166
# 2023_04_02
# Assignment_03_01


#Import Libraries
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


def confusion_matrix(y_true, y_pred, n_classes=10):
    # Converting labels to encoded format
    if len(y_pred.shape) > 1:
        y_pred = np.argmax(y_pred, axis=1)
    if len(y_true.shape) > 1:
        y_true = np.argmax(y_true, axis=1)
    
    #print(y_true.shape)
    #print(y_pred.shape)
    
    # Computing the confusion matrix for a set of predictions
    confusion_matrx = np.zeros((n_classes, n_classes), dtype=np.int32)
    for i in range(len(y_true)):
        pred_class = y_pred[i]
        true_class = y_true[i]
        confusion_matrx[true_class][pred_class] = confusion_matrx[true_class][pred_class]+ 1
    return confusion_matrx
     

def train_nn_keras(X_train, Y_train, X_test, Y_test, epochs=1, batch_size=4):
    

    tf.keras.utils.set_random_seed(5368) # do not remove this line
    #CNN Layers
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Conv2D(filters=8, kernel_size=(3,3), strides=(1,1), padding='same', activation='relu', input_shape=(28,28,1), kernel_regularizer=tf.keras.regularizers.L2(0.0001)))
    model.add(tf.keras.layers.Conv2D(16, (3,3), strides=(1,1), padding='same', activation='relu', kernel_regularizer=tf.keras.regularizers.L2(0.0001)))
    model.add(tf.keras.layers.MaxPooling2D((2,2)))
    model.add(tf.keras.layers.Conv2D(32, (3,3), strides=(1,1), padding='same', activation='relu', kernel_regularizer=tf.keras.regularizers.L2(0.0001)))
    model.add(tf.keras.layers.Conv2D(64, (3,3), strides=(1,1), padding='same', activation='relu', kernel_regularizer=tf.keras.regularizers.L2(0.0001)))
    model.add(tf.keras.layers.MaxPooling2D((2,2)))
    #Dense and Flatten Layers
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(512, activation='relu', kernel_regularizer=tf.keras.regularizers.L2(0.0001)))
    model.add(tf.keras.layers.Dense(10, activation='linear', kernel_regularizer=tf.keras.regularizers.L2(0.0001)))
    model.add(tf.keras.layers.Activation('softmax'))
    

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
   
    hist = model.fit(X_train, Y_train, epochs=epochs, batch_size=batch_size, validation_split=0.2)
    
    model.save("model.h5")

    # Evaluating model on the test set
    Y_pred = model.predict(X_test)
    Y_predcls = np.argmax(Y_pred, axis=1)
    Y_true = np.argmax(Y_test,axis = 1)
    
    # Computing the confusion matrix on test set
    confusion_matrx = confusion_matrix(Y_true,Y_predcls )
    
    
 # Plot the confusion matrix as heatmap
    plt.matshow(confusion_matrx)
    plt.colorbar()
    
    plt.xlabel('Predicted label')
    plt.ylabel('Actual label')
    plt.savefig('confusion_matrix.png')
    
    #print(f"{Y_pred = !r}")
    
    return [model, hist, confusion_matrx, Y_predcls]
