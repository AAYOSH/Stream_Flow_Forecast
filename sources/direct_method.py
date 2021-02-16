import os 
import pandas as pd
import numpy as np
from datetime import datetime,timedelta
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from math import sqrt
from numpy import split
from numpy import array
from pandas import read_csv
from sklearn.metrics import mean_squared_error
from matplotlib import pyplot
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import LSTM

def to_supervised(train, n_input, n_out=7):
        X, y = list(), list()
        in_start = 0
        # step over the entire history one time step at a time
        for _ in range(len(train)):
            # define the end of the input sequence
            in_end = in_start + n_input
            out_end = in_end + n_out
            # ensure we have enough data for this instance
            if out_end <= len(train):
                x_input = train[in_start:in_end, 0]
                x_input = x_input.reshape((len(x_input), 1))
                X.append(x_input)
                y.append(train[in_end:out_end, 0])
            # move along one time step
            in_start += 1
        return np.array(X), np.array(y)
    
    
def build_model(train, test, n_input,n_output,epochs,batch_size,n_units):
    # prepare data
    train_x, train_y = to_supervised(train, n_input,n_output)
    test_x,test_y = to_supervised(test, n_input,n_output)
    # define parameters
    verbose = 1
    n_timesteps, n_features, n_outputs = train_x.shape[1], train_x.shape[2], train_y.shape[1]
    # define model
    model = Sequential()
    model.add(LSTM(n_units, activation='relu', input_shape=(n_timesteps, n_features)))
    model.add(Dense(100, activation='relu'))
    model.add(Dense(n_outputs))
    model.compile(loss='mse', optimizer='adam')
    # fit network
    history = model.fit(train_x, train_y, epochs=epochs, batch_size=batch_size, verbose=verbose, validation_data = (test_x, test_y))
    return model,history


def evaluate_modelo(previsao,esperado):
   
    # calcula score para cada dia do horizonte de 7
    scores = list()
    for i in range(esperado.shape[1]):
        # calculate mse
        mse = mean_squared_error(esperado[:, i], previsao[:, i])
        # calculate rmse
        rmse = sqrt(mse)
        # store
        scores.append(rmse)
        
        
    # calcula score geral
    s = 0
    for row in range(esperado.shape[0]):
        for col in range(esperado.shape[1]):
            s += (esperado[row, col] - previsao[row, col])**2
            
    score = sqrt(s / (esperado.shape[0] * esperado.shape[1]))
    return score, scores

def series_forecast(model,amostra,input_len):
    
    '''
    apenas devolve output de 7 passos
    '''
    y_hat =  list()
    
    X_input = amostra.ravel()
    for j in range(input_len):
        y_ = model.predict(X_input.reshape(1,input_len,1))[0][0]
        X_input = np.append(X_input[1:],y_)
        y_hat.append(y_)
        
    return y_hat

def direct_method(train,test,n_input,n_output,epochs,batch_size,n_units):
    
    train = np.array(train).reshape(-1,1)
    test = np.array(test).reshape(-1,1)
    
    
    scaler = MinMaxScaler(feature_range=(0, 1))
    train_scaled = scaler.fit_transform(train)
    test_scaled = scaler.fit_transform(test)
    
    modelo,historico = build_model(train_scaled,test_scaled,n_input,n_output,epochs,batch_size,n_units)
    

    return modelo,historico


if __name__ == '__main__':

    df = pd.read_excel('../data/Vazões_Diárias_1931_2018.xlsx')
#===================================================================
# Cleaning / formating the dataframe
    x = df.copy()
    columns = x.loc[4,:]
    x.columns  =  np.concatenate([['DATA'],columns[1:]])
    x = x[6:]
    x.index = x['DATA']
    x = x.drop('DATA',axis = 1)
    x.index = pd.date_range('1931-01-01','2018-12-31')

    data = x.loc[:,'A. VERMELHA (18)'] # > chosing the Água Vermelha Power plant stream flow historic
# ===================================================================
# Selecting the dates for train/test
    project_data = data[(data.index > datetime(1999,12,31)) & (data.index < datetime(2010,1,1))].astype('float32')
    train = project_data.iloc[:int(len(project_data)*0.8)]
    test = project_data.iloc[int(len(project_data)*0.8):]

#=====================================================================
# with the hyperparameters values found in a previous step not shown here, train and test model
    ########## parameters
    n_input = 25
    n_output = 7
    epochs = 30
    batch_size = 2
    n_units = 100
    #####################

# ====================================================================
# train dataset
    modelo,historico = direct_method(train,test,n_input,n_output,epochs,batch_size,n_units)
    train = np.array(train).reshape(-1,1)
    scaler = MinMaxScaler(feature_range=(0, 1))
    train_scaled = scaler.fit_transform(train)

    X,Y = to_supervised(train_scaled,n_input,n_output)
    y_hat = modelo.predict(X)

    y_hat = scaler.inverse_transform(y_hat)
    Y = scaler.inverse_transform(Y)

    score_,scores_ = evaluate_modelo(y_hat,Y)

    print(f"Performance train set direct method {score_,scores_}")

#===========================================================================
# test dataset

    test = np.array(test).reshape(-1,1)
    scaler = MinMaxScaler(feature_range=(0, 1))
    test_scaled = scaler.fit_transform(test)


    X,Y = to_supervised(test_scaled,n_input,n_output)
    y_hat = modelo.predict(X)

    y_hat = scaler.inverse_transform(y_hat)
    Y = scaler.inverse_transform(Y)

    score_,scores_ = evaluate_modelo(y_hat,Y)

    print(f"Performance test set direct method  {score_,scores_}")