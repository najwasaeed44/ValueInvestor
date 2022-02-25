import pmdarima as pm
import numpy as np
import pandas as pd
import tensorflow as tf
from matplotlib import pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau



def train_test_split(df):
    test = df[df.Date.astype(str).str.contains('2021')]
    train = df[df.Date.astype(str).str.contains("2021") == False]
    return train, test


def arima_model(data_name, train, test):
    arima_model = pm.auto_arima(train.difference, start_p=1, start_q=1, test='adf', max_p=5, max_q=5, 
                      m=1, d=None, seasonal=False, start_P=0, D=1, trace=False, error_action='ignore',  
                      suppress_warnings=True, stepwise=False)

    pred = arima_model.predict(len(test))
    arima_model_result = test[['difference']]
    arima_model_result['Forecast_ARIMA'] = pred

    plt.figure(figsize=(16, 5))
    plt.plot(train.difference, label='Train', linewidth=1.5)
    plt.plot(arima_model_result.difference, label='Test', linewidth=1.5)
    plt.plot(arima_model_result.Forecast_ARIMA, label='Predictions', linewidth=1.5)
    plt.title(f'Actual  VS. Predicted {data_name} Price for ARIMA model ', fontsize=20)
    plt.legend();
    mae = np.mean(np.abs(arima_model_result.Forecast_ARIMA - arima_model_result.difference)) 
    model_name = f"{data_name} ARIMA Model"
    return model_name, mae

def sarima_model(data_name, train, test):
    sarima_model = pm.auto_arima(train.difference, start_p=1, start_q=1, test='adf', max_p=5, max_q=5, 
                      m=1, d=None, seasonal=True, start_P=0, D=1, trace=False, error_action='ignore',  
                      suppress_warnings=True, stepwise=False)

    pred = sarima_model.predict(len(test))
    sarima_model_result = test[['difference']]
    sarima_model_result['Forecast_SARIMA'] = pred

    plt.figure(figsize=(16, 5))
    plt.plot(train.difference, label='Train', linewidth=1.5)
    plt.plot(sarima_model_result.difference, label='Test', linewidth=1.5)
    plt.plot(sarima_model_result.Forecast_SARIMA, label='Predictions', linewidth=1.5)
    plt.title(f'Actual  VS. Predicted {data_name} Price for SARIMA model ', fontsize=20)
    plt.legend();
    mae = np.mean(np.abs(sarima_model_result.Forecast_SARIMA - sarima_model_result.difference))  
    model_name = f"{data_name} SARIMA Model"
    return model_name, mae

def arimax_model(data_name, train, test):
    exogenous_features = ['Open_mean_lag3', 'Open_mean_lag7','Open_mean_lag30', 'Open_std_lag3', 
                          'Open_std_lag7', 'Open_std_lag30', 'High_mean_lag3', 'High_mean_lag7', 
                          'High_mean_lag30', 'High_std_lag3', 'High_std_lag7', 'High_std_lag30', 
                          'Low_mean_lag3', 'Low_mean_lag7', 'Low_mean_lag30', 'Low_std_lag3', 
                          'Low_std_lag7', 'Low_std_lag30', 'Vol._mean_lag3', 'Vol._mean_lag7', 
                          'Vol._mean_lag30', 'Vol._std_lag3', 'Vol._std_lag7', 'Vol._std_lag30', 
                          'Change %_mean_lag3', 'Change %_mean_lag7', 'Change %_mean_lag30', 'Change %_std_lag3',
                          'Change %_std_lag7', 'Change %_std_lag30', "month", "week", "day", "day_of_week"]
    
    arimax_model = pm.auto_arima(train.difference, exogenous=train[exogenous_features], trace=False, error_action="ignore", suppress_warnings=True)
    arimax_model.fit(train.difference, exogenous=train[exogenous_features])

    forecast = arimax_model.predict(n_periods=len(test), exogenous=test[exogenous_features])
    test["Forecast_ARIMAX"] = forecast
    plt.figure(figsize=(16, 5))

    plt.plot(train.difference, label='Train', linewidth=1.5)
    plt.plot(test.difference, label='Test', linewidth=1.5)
    plt.plot(test.Forecast_ARIMAX, label='Predictions', linewidth=1.5)    
    plt.title(f'Actual  VS. Predicted {data_name} Price for ARIMAX model ', fontsize=20)
    plt.legend();
    mae = np.mean(np.abs(test.Forecast_ARIMAX - test.difference)) 
    model_name = f"{data_name} ARIMAX Model"
    return model_name, mae

def Lookback_data(df, lookback):
    data_raw = df.to_numpy()
    data = []
    for index in range(len(data_raw) - lookback): 
        data.append(data_raw[index: index + lookback + 1])
    data = np.array(data)
    
    X = data [:, :-1, :].reshape(len(data), 5)
    y = data[:, -1, 0].reshape(-1,1)
    
    
    df_featured = pd.DataFrame(X)
    df_featured['y'] = y
    df_featured.set_index(df.index[lookback:], inplace = True)
    return df_featured

def train_valid_test_splt(df):
    train, test = train_test_split(df)
    train_data = Lookback_data(train[['difference']], 5)
    test_data = Lookback_data(test[['difference']], 5)
    
    ratio = len(train_data) * 0.8
    train_dataset = train_data.iloc[:int(ratio)]
    valid_dataset = train_data.iloc[int(ratio):]
    
    scaler = MinMaxScaler(feature_range=(0, 1))

    train_scaled = pd.DataFrame(scaler.fit_transform(train_dataset[train_dataset.columns]))
    val_scaled = pd.DataFrame(scaler.transform(valid_dataset[valid_dataset.columns]))
    test_scaled = pd.DataFrame(scaler.transform(test_data[test_data.columns]))
    
    train_scaled.columns = ['day1', 'day2', 'day3', 'day4', 'day5', 'y']
    val_scaled.columns = ['day1', 'day2', 'day3', 'day4', 'day5', 'y']
    test_scaled.columns = ['day1', 'day2', 'day3', 'day4', 'day5', 'y']
    
    X_train = train_scaled.drop('y', axis=1).values
    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
    y_train = train_scaled['y'].values


    X_val = val_scaled.drop('y', axis=1).values
    X_val = np.reshape(X_val, (X_val.shape[0], X_val.shape[1], 1))
    y_val = val_scaled['y'].values

    X_test = test_scaled.drop('y', axis=1).values
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
    y_test = test_scaled['y'].values
    
    return X_train, y_train, X_val, y_val, X_test, y_test


def LSTM(X_train, y_train, X_val, y_val, X_test, y_test, data_name):
    tf.random.set_seed(1)
    print('\n')
    print("#" * 50)
    print('\n\n SRART THE TRAINING\n\n')
    print("#" * 50)
    print('\n')

    # 1) Create the model
    model = tf.keras.Sequential([
      tf.keras.layers.LSTM(units=100, return_sequences=True, activation='tanh', input_shape=(X_train.shape[1], 1)),
      tf.keras.layers.Dropout(0.15),
      tf.keras.layers.LSTM(units=100), 
      tf.keras.layers.Dense(units=1)])

    # 2) Compile the model
    model.compile(loss=tf.keras.metrics.mae, optimizer=tf.keras.optimizers.Adam())

    # 3) Fit the model
    callbacks = [
               EarlyStopping(monitor='val_loss', mode='min',patience=10,  verbose=1),
              #  ReduceLROnPlateau(factor=0.1, patience=3, min_lr=0.00001, min_delta = 0.00000000001, verbose=1),
               ModelCheckpoint(f'{data_name}_LSTM_MODEL.h5', verbose=1, save_best_only=True)]
    hist = model.fit(X_train, y_train, epochs =1000, batch_size=16, validation_data=(X_val, y_val), callbacks=[callbacks])
    print('\n')
    print("#" * 50)  
    print('\n\n TRAINING IS FINISHED\n\n')
    print("#" * 50)

    plt.figure(figsize=(16, 5))
    plt.plot(hist.history['loss'], label='Train loss', linewidth=1.5)
    plt.plot(hist.history['val_loss'], label='Valid loss', linewidth=1.5)
    plt.title('Train VS. Valid Loss', fontsize=20)
    plt.legend();
    scaler = MinMaxScaler(feature_range=(0,1))
    predictions = model.predict(X_test)
    y_test_scaled = y_test.reshape(-1,1)
    scaler.fit(y_test_scaled)
    prediction_inverse = scaler.inverse_transform(predictions)

    fig, ax = plt.subplots(figsize=(16, 5))
    ax.plot(y_test_scaled, label='Original price', linewidth=1.5)
    plt.plot(prediction_inverse, label='Predicted price', linewidth=1.5)
    plt.title('Original price VS Predicted price', fontsize=20)
    plt.legend();
    mae = np.mean(np.abs(prediction_inverse - y_test_scaled)) 
    model_name = f"{data_name} LSTM Model"
    return model_name, mae

    

def scores(scores_dict):
    return pd.DataFrame(scores_dict)