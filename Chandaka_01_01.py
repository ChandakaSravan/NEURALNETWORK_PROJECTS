# Chandaka, Sravan
# 1002_059_166
# 2023_02_27
# Assignment_01_01

#Import libraries
import numpy as np


#Sigmoid Calculation
def sigmoid(z):
    return 1 / (1 + np.exp(-z))


#Initilaizing weights 
def weights_initiate(layers,seed,X_train):
    rows,cols = X_train.shape
    matrix_wt = []

    #Randomizing weights generation
    for i in range(len(layers)):
        if i==0:
            np.random.seed(seed)
            m_1 = np.random.randn(layers[i],rows+1)
            matrix_wt.append(m_1)
        else:
            np.random.seed(seed)
            m_1 = np.random.randn(layers[i],layers[i-1]+1)
            matrix_wt.append(m_1)
    return matrix_wt

def Activation_fn(mtrx_input):
    for i in range(len(mtrx_input)):
        mtrx_input[i] = sigmoid(mtrx_input[i])
    return mtrx_input


#Forward propagation function to predict output
def forward_prop(X_train,layers,matrix_wt):
    current_state = X_train.copy()
    for i in range(len(layers)):
        rows,cols = current_state.shape
        bias = np.ones(cols)
        current_state = np.vstack((bias,current_state))
        curr_matrix = np.dot(matrix_wt[i],current_state)
        curr_matrix = Activation_fn(curr_matrix)
        current_state = curr_matrix
    return current_state

import copy

#Backward propagation to adjust weights of the weight matrix
def backward_prop(X,Y,layers,matrix_wt,h,alpha):
    ext_matrix = copy.deepcopy(matrix_wt)
    new_matrix_wt = []
    
    for i in range(len(matrix_wt)):
        for rows in range(len(matrix_wt[i])):
            for cols in range(len(matrix_wt[i][rows])):
                original_value = matrix_wt[i][rows][cols]
                #incrementing the value by h
                curr_value = original_value+h
                matrix_wt[i][rows][cols] = curr_value
                predict = forward_prop(X,layers,matrix_wt)
                a = np.mean((Y - predict)**2)
                matrix_wt[i][rows][cols] = original_value-h
                #decrementing the value by h
                predict = forward_prop(X,layers,matrix_wt)
                #mse
                b = np.mean((Y - predict)**2)
                #Calculating the error 
                err_calc = (a-b)/(2*h)
                nw_wght = original_value-(alpha*err_calc)
                ext_matrix[i][rows][cols] = nw_wght
                matrix_wt[i][rows][cols] = original_value
                
    return ext_matrix

def multi_layer_nn(X_train,Y_train,X_test,Y_test,layers,alpha,epochs,h=0.00001,seed=2):
    mse = []
    matrix_wt = weights_initiate(layers,seed,X_train)
#     print(matrix_wt)
    for i in range(epochs):
        matrix_wt = backward_prop(X_train,Y_train,layers,matrix_wt,h,alpha)
#         print(matrix_wt)
        predict = forward_prop(X_test,layers,matrix_wt)
#         print(predict)
        err_calc = np.mean((Y_test - predict)**2)
#         print(err_calc)
        mse.append(err_calc)
#     print(mse)
    mse = np.array(mse,dtype=list)
    #predicting the output
    test_predict = forward_prop(X_test,layers,matrix_wt)
    #returning the result
    return matrix_wt,mse,test_predict
