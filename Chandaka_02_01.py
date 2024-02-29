# Chandaka, Sravan
# 1002_059_166
# 2023_03_20
# Assignment_02_01

import numpy as np
import tensorflow as tf

# Calculating the loss for the given predictions
def calc_loss(target_y, prediction, computeloss):
    x=computeloss(target_y, prediction)
    return tf.reduce_mean(x)

# Activation functions and loss functions.
def activation_loss_functions(activation):
    if activation == 'relu':
        return tf.nn.relu
    elif activation == 'sigmoid':
        return tf.sigmoid
    elif activation == 'linear':
        return tf.identity
    elif activation == 'mse':
        return mean_squared_error
    elif activation == 'svm':
        return svm
    elif activation == 'cross_entropy':
        return crossentropy_loss


def mean_squared_error(target_y, predict_y):
    return tf.reduce_mean(tf.square(target_y - predict_y))


def svm(target_y, predict_y):
    loss_calc = tf.maximum(0.0, 1 - target_y * predict_y)
    return tf.reduce_mean(loss_calc)


def crossentropy_loss(target_y, predict_y):
    softmax_preds = tf.nn.softmax(predict_y)
    crossentropy_loss = -tf.reduce_sum(target_y * tf.math.log(softmax_preds), axis=1)
    mean_crossentropy_loss = tf.reduce_mean(crossentropy_loss)
    return mean_crossentropy_loss


activation_fn = {}
for activation in ['relu', 'sigmoid', 'linear']:
    activation_fn[activation] = activation_loss_functions(activation)

loss_fn = {}
for activation in ['mse', 'svm', 'cross_entropy']:
    loss_fn[activation] = activation_loss_functions(activation)



# Based on given split_range it splits the data into training and validation sets 


def split_data(X_train, Y_train, split_range=[0.2, 0.7]):
    # Validation set to calculate the start and end indices 
    start_index = int(split_range[0] * len(X_train))
    end_index = int(split_range[1] * len(X_train))

    # Splits the data into training and validation sets
    X_train_new = np.concatenate((X_train[:start_index], X_train[end_index:]), axis=0)
    Y_train_new = np.concatenate((Y_train[:start_index], Y_train[end_index:]), axis=0)
    X_val = X_train[start_index:end_index]
    Y_val = Y_train[start_index:end_index]

    return X_train_new, Y_train_new, X_val, Y_val



# Generating batches with the specified batch_size.

def generate_batches(X, y, batch_size=32):
    for i in range(0, X.shape[0], batch_size):
        yield X[i:i+batch_size], y[i:i+batch_size]
    # if there's any data left, yield it
    if X.shape[0] % batch_size != 0:
        yield X[-(X.shape[0] % batch_size):], y[-(X.shape[0] % batch_size):]

#Initilaizing weights 
def intializing_weights(Xtrain_dim, layers, seed):
    matrix_wt = []
    weights=None
    num_layers = len(layers)
    #Randomizing weights generation
    for i in range(num_layers):
        np.random.seed(seed)
        if i == 0:
            input_dim = Xtrain_dim
        else:
            input_dim = layers[i-1]
        output_dim = layers[i]
        weight_matrix = np.random.randn(input_dim+1, output_dim).astype(np.float32)
        if weights is not None:
            weight_matrix = weights[i]
        matrix_wt.append(tf.Variable(weight_matrix))
    return matrix_wt

#Forward propagation function to predict output
def forward_prop(weights, X, activation_fn_loss, activations_names):
    # Adding a column of ones for bias.
    X = tf.concat([tf.ones((tf.shape(X)[0], 1)), X], axis=1)
    current_state = []
    # Iterating through the layers, by activation functions and updating list
    for i in range(len(weights)):
        Z = tf.matmul(X if i == 0 else current_state[-1], weights[i])
        act_loss = activation_fn[activations_names[i]](Z)
        if i != len(weights) - 1:
            act_loss = tf.concat([tf.ones((tf.shape(act_loss)[0], 1)), act_loss], axis=1)
        #print(act_loss)
        current_state.append(act_loss)
    
    return current_state


def fit_grad(X_train,Y_train,alpha,epochs,batch_size,weights,activations,X_val, Y_val,loss):
    error=[]
    loss_calc=[]
    prediction=[]
     # Training for specific number of epochs.
    for epoch in range(epochs):
            for X_trainbatch, y_trainbatch in generate_batches(X_train, Y_train, batch_size):
                 # Calculating the gradients of the loss 
                with tf.GradientTape(persistent=True) as tape:
                    tape.watch(weights)
                    predict = forward_prop( weights, X_trainbatch, activation_fn, activations)
                    prediction= predict[-1]
                    calcloss_train_gradient = calc_loss(y_trainbatch, prediction, loss_fn[loss])
                    t_grad = tape.gradient(calcloss_train_gradient, weights)
                    
                    
         # Updating the weights using gradients and learning rate alpha.
                for layer in range(len(weights)):
                    gradient_current =  t_grad[layer]
                    weights[layer].assign_sub(alpha * gradient_current)
            predict_val = forward_prop(weights, X_val, activation_fn, activations)
            predict_val = predict_val[-1]
            val_loss = calc_loss(Y_val, predict_val, loss_fn[loss])
            # Add loss for the current epoch to the loss_calc list.
            loss_calc.append(val_loss)
            error =list(loss_calc)
    return error


def multi_layer_nn_tensorflow(X_train, Y_train, layers, activations, alpha, batch_size, epochs=1, loss="svm",
                              validation_split=[0.8, 1.0], weights=None, seed=2):

    errors=[]
    Xtrain_dim = X_train.shape[1]

     # Initializing the weights if not provided.
    if weights is None:
        weights = intializing_weights(Xtrain_dim, layers, seed)

    X_train, Y_train, X_val, Y_val = split_data(X_train, Y_train, split_range=validation_split)
    
    errors = fit_grad(X_train,Y_train,alpha,epochs,batch_size,weights,activations,X_val, Y_val,loss)

    #predicting the output
    output = forward_prop(weights, X_val, activation_fn, activations)
    test_predict = output[-1]
    #print(weights)
    #print(errors)
    # Return the  weights, errors, and the final test predictions

    return [weights, errors, test_predict]
