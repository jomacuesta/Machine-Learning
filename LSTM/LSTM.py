# -*- coding: utf-8 -*-
"""
@author: Jose Manuel Cuesta
"""

##-----------------------------LIBRERIAS

import pandas as pd
#import os
#import glob
import numpy as np
#from datetime import datetime
import matplotlib.pyplot as plt
#import seaborn as sns
#import tensorflow as tf
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dense,BatchNormalization,Dropout 
from tensorflow.keras.models import Sequential
#from tensorflow.keras import regularizers
#from sklearn.preprocessing import StandardScaler
#from tensorflow.keras.optimizers import SGD
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error


##------------------------------FUNCIONES
    
def LSTM_model(scaled_train_data):
    
    x_train = []
    y_train = []
    
    for i in range(60, len(scaled_train_data)): #60 registros anteriores (60 timesteps)
    
        x_train.append(scaled_train_data[i-60:i,0])
        y_train.append(scaled_train_data[i,0])
    
    x_train = np.array(x_train)
    y_train = np.array(y_train)
    x_train = np.reshape(x_train, (x_train.shape[0],x_train.shape[1],1)) # 3D [samples, timesteps, features]
    
    #Arquitectura del modelo 
    lstm_model = Sequential()
    lstm_model.add(LSTM(128, return_sequences = True, activation='relu', input_shape=(x_train.shape[1], 1)))
    lstm_model.add(LSTM(32, return_sequences=False, activation='relu'))
    lstm_model.add(Dense(16))
    lstm_model.add(Dense(1))
    lstm_model.compile(optimizer='adam', loss='mse')
    lstm_model.fit(x_train,y_train, batch_size = 1, epochs = 5) #Valido para probar - combinacion estándar (batch = 32, epochs = 50)
    #history = lstm_model.fit(x_train,y_train, epochs=20, batch_size=16, validation_data=(test_X, test_y), verbose=2, shuffle=False)
    #plot_results(history)
    
    return lstm_model



def plot_results(history):
    
    # summarize history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper right')
    plt.show()
    


def run_Forecasting(lstm_model,scaled_train_data,test_data):
    
    """ Forecast Real ("Desconocido" el conjunto de test) , Se van añadiendo y usando las predicciones"""
    
    n_features=1
    n_input = 60

    lstm_predictions_scaled = list()
    batch = scaled_train_data[-n_input:] #Generamos el último batch antes de la predicción forecast
    current_batch = batch.reshape((1, n_input, n_features))

    for i in range(len(test_data)):   
        
        lstm_pred = lstm_model.predict(current_batch)[0]
        lstm_predictions_scaled.append(lstm_pred) 
        #En el siguiente paso, vamos moviendo el batch de 1 en 1 y añadimos la predicción realizada 
        current_batch = np.append(current_batch[:,1:,:],[[lstm_pred]],axis=1) 

    print("Predicciones realizadas sin invertir")
    
    return lstm_predictions_scaled


def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
    
    """ Pivotar la serie de datos si incluye más columnas exógenas """

    n_vars = 1 if type(data) is list else data.shape[1]
    df = pd.DataFrame(data)
    cols, names = list(), list()
    # input sequence (t-n, ... t-1)
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
        names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]
    # forecast sequence (t, t+1, ... t+n)
    for i in range(0, n_out):
        cols.append(df.shift(-i))
        if i == 0:
            names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
        else:
            names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]
    # put it all together
    agg = pd.concat(cols, axis=1)
    agg.columns = names
    # drop rows with NaN values
    if dropnan:
        agg.dropna(inplace=True)
        
    return agg


def frame_series(scaled): #Como parametros el dataset completo escalado

    """ Prepara la tabla de datos con los datos del dia anterior para cada una de las variables """

    df_table = series_to_supervised(scaled, 1, 1)
    df_table.drop(df_table.columns[[8,9,10,11,12,13]], axis=1, inplace=True) # Ejemplo para 6 columnas exógenas y 1 columna target
    # split into train and test sets
    values = df_table.values
    n_train_time = 365*24
    train = values[:n_train_time, :]
    test = values[n_train_time:, :]
    # split into input and outputs
    train_X, train_y = train[:, :-1], train[:, -1]
    test_X, test_y = test[:, :-1], test[:, -1]
    # 3D [samples, timesteps, features]
    train_X = train_X.reshape((train_X.shape[0], 1, train_X.shape[1])) #Si lo hacemos de este modo, a diferencia de la función LSTM_model debemos aumentar el batch_size >> 1 (1 timestep)
    test_X = test_X.reshape((test_X.shape[0], 1, test_X.shape[1]))
    
    return train_X, train_y, test_X, test_y  #Usar en run_predictions_test

def run_predictions_test(model,test_X,test_y): #Función con variable exógenas conocidas
    
    # Make a prediction
    yhat = model.predict(test_X)
    test_X = test_X.reshape((test_X.shape[0], 7)) 
    # Invert scaling for forecast
    inv_yhat = np.concatenate((yhat, test_X[:, -6:]), axis=1)
    inv_yhat = scaler.inverse_transform(inv_yhat)
    inv_yhat = inv_yhat[:,0]
    # Invert scaling for actual
    test_y = test_y.reshape((len(test_y), 1))
    inv_y = np.concatenate((test_y, test_X[:, -6:]), axis=1)
    inv_y = scaler.inverse_transform(inv_y)
    inv_y = inv_y[:,0]
    rmse = np.sqrt(mean_squared_error(inv_y, inv_yhat))
    print('Test RMSE: %.3f' % rmse)
    
    return inv_yhat, inv_y

""" Predecir usando 60 timesteps con valores X conocidos. (Predicción directa)

def run_predictions_test(regressor,inputs):
    
    # Preparar X_test con 60 timesteps
    X_test = []
    for i in range(60,len(inputs)):
        X_test.append(inputs[i-60:i,0])
    X_test = np.array(X_test)
    X_test = np.reshape(X_test, (X_test.shape[0],X_test.shape[1],1))
    predicted_stock_price = regressor.predict(X_test)
    #predicted_stock_price = sc.inverse_transform(predicted_stock_price)
    
    return predicted_stock_price

"""    


##------------------------------MAIN

if __name__ == "__main__":
    
    """
    Este Script representa la plantilla para un Forecast empleando una red RNN - LSTM univariable (1 variable en el tiempo).
    Se añaden funciones destinadas a la creación de módelos LSTM con variables exógenas y previamente conocidas : series_to_supervised , frame_series y run_Predictions_test

    """
        
    #-------Se escalan los datos
    values = "Fichero de datos"
    n_train_index = "Número de registros para separar la serie de datos"
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(values)
    train_data = scaled_data[:n_train_index, :]
    test_data = scaled_data[n_train_index:, :] 
    
    #-------Creación del modelo
    model = LSTM_model(train_data)
    #-------Forecast
    lstm_predictions_scaled = run_Forecasting(model,train_data,test_data)
    #-------Deshacemos el efecto de MinMaxScaler
    lstm_predictions = scaler.inverse_transform(lstm_predictions_scaled)
