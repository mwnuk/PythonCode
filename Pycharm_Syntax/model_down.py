import pandas as pd
#from pandas_datareader import data as web
from pandas import read_csv
from pandas import concat
import os.path
import datetime
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from keras.utils.np_utils import to_categorical
import numpy
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM


#mypath="C:\\Users\\mwnuk\\Dropbox\\Quotes\\"
#myModelpath='C:\\Users\\mwnuk\\Dropbox\\ModelsDown\\'
#settingsPath='C:\\Users\\mwnuk\\Dropbox\\Settings\\'

mypath="./data/Quotes/"
myModelpath='./data/ModelsDown/'
settingsPath='./data/Settings/'

#####################################################################
# Stochastic Oscilator %K
def STOK(close, low, high, n):
    STOK = ((close - pd.Series.rolling(low, n).min()) / (
    pd.Series.rolling(high, n).max() - pd.Series.rolling(low, n).min())) * 100
    return STOK


# Stochastic Oscilator %D
def STOD(close, low, high, n):
    STOK = ((close - pd.Series.rolling(low, n).min()) / (
    pd.Series.rolling(high, n).max() - pd.Series.rolling(low, n).min())) * 100
    STOD = pd.Series.rolling(STOK, 3).mean()
    return STOD


#####################################################################
def RSI(series, period):
    delta = series.diff().dropna()
    u = delta * 0
    d = u.copy()
    u[delta > 0] = delta[delta > 0]
    d[delta < 0] = -delta[delta < 0]
    u[u.index[period - 1]] = np.mean(u[:period])  # first value is sum of avg gains
    u = u.drop(u.index[:(period - 1)])
    d[d.index[period - 1]] = np.mean(d[:period])  # first value is sum of avg losses
    d = d.drop(d.index[:(period - 1)])
    # rs = pd.stats.moments.ewma(u, com=period-1, adjust=False) / \
    #      pd.stats.moments.ewma(d, com=period-1, adjust=False)
    rs = pd.Series.ewm(u, com=period - 1, min_periods=0, adjust=False, ignore_na=False).mean() / \
         pd.Series.ewm(d, com=period - 1, min_periods=0, adjust=False, ignore_na=False).mean()
    return 100 - 100 / (1 + rs)


#####################################################################

def getData(item):
    start = datetime.datetime(2015, 12, 21)
    end = datetime.datetime.now()
    # print( start,end)
    file_path = mypath + item + '.csv'
    if not os.path.exists(file_path):
        print("Empty set")
        df = web.DataReader(item, 'yahoo', start, end)
        df = df.round(2)

        df.to_csv(mypath + item + '.csv')
        # df['Date']=pd.to_datetime(df['Date']) #important for sorting
        # df.set_index("Date",inplace=True)
        df.index.name = 'Date'
    else:
        print("Read from file")
        df = read_csv(file_path)
        df['Date'] = pd.to_datetime(df['Date'])  # important for sorting
        df.set_index("Date", inplace=True)


        return df


def cleanData(df):
    # drop Close and Volume, keep Adj Close
    df1 = df.drop('Close', 1)
    df1 = df1.drop('Volume', 1)
    df1.rename(columns={'Adj Close': 'Close', 'oldName2': 'newName2'}, inplace=True)
    return df1


#####################################################################
# Engineer features
def engFeatures(df):
    df['%K'] = STOK(df['Close'], df['Low'], df['High'], 14)
    df['%D'] = STOD(df['Close'], df['Low'], df['High'], 14)
    df['Avg5'] = pd.Series.rolling(df['Close'], 5).mean()
    df['Avg10'] = pd.Series.rolling(df['Close'], 10).mean()
    df['RSI14'] = RSI(df['Close'], 14)
    df['RSI7'] = RSI(df['Close'], 7)
    df.sort_index(ascending=False, inplace=True)
    #df['Rise'] = (pd.Series.rolling(df['Close'], 5).max() - df['Close'] > df['Close'] * 0.04) * 1
    df['Fall'] = (  df['Close']-pd.Series.rolling(df['Close'],5).min() > df['Close']*0.03)*1
    df.sort_index(ascending=True, inplace=True)
    return df


#####################################################################
# convert series to supervised learning
def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
    n_vars = 1 if type(data) is list else data.shape[1]
    df = pd.DataFrame(data)
    cols, names = list(), list()
    # input sequence (t-n, ... t-1)
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
        names += [('var%d(t-%d)' % (j + 1, i)) for j in range(n_vars)]
    # forecast sequence (t, t+1, ... t+n)
    for i in range(0, n_out):
        cols.append(df.shift(-i))
        if i == 0:
            names += [('var%d(t)' % (j + 1)) for j in range(n_vars)]
        else:
            names += [('var%d(t+%d)' % (j + 1, i)) for j in range(n_vars)]
    # put it all together
    agg = concat(cols, axis=1)
    agg.columns = names
    # drop rows with NaN values
    if dropnan:
        agg.dropna(inplace=True)
    return agg



#####################################################################
def rescale(df):
    df.dropna(inplace=True)
    values = df.values
    # ensure all data is float
    values = values.astype('float')
    # drop columns to simplify test
    timesteps = 3
    features = values.shape[1] - 1
    # normalize features
    scaler = MinMaxScaler()
    # scaled = scaler.fit_transform(values)
    scaled = values  # scaler.fit_transform(values)
    # frame as supervised learning
    reframed = series_to_supervised(scaled, timesteps, 1)
    # drop columns we don't want to predict
    # reframed.drop(reframed.columns[[10,11,12,13,14,15,16,17,18,19,20]], axis=1, inplace=True)
    reframed = pd.DataFrame(reframed)
    return reframed


#####################################################################

def buildmodel():
    model = Sequential()

    # this is good .92
    model.add(LSTM(20, return_sequences=True, input_shape=(train_X.shape[1], train_X.shape[2])))
    model.add(LSTM(20))
    model.add(Dense(8, input_dim=3, activation='relu'))
    model.add(Dense(2, activation='sigmoid'))

    '''#this is good .89
    model.add(LSTM(20, input_shape=(train_X.shape[1], train_X.shape[2])))
    model.add(Dense(8, input_dim=3, activation='relu'))
    model.add(Dense(2, activation='sigmoid'))'''

    '''#this is good .88
    model.add(LSTM(50, input_shape=(train_X.shape[1], train_X.shape[2])))
    model.add(Dense(8, input_dim=3, activation='relu'))
    model.add(Dense(2, activation='sigmoid'))'''

    # Dense expects a 2-dimensional input (batch_size, features),
    # whereas the output of LSTM with return_sequences is 3 dimensional (batch_size, timesteps, features).

    # stacked --good acc =0.843
    '''model.add(LSTM(4, input_shape=(43, 1)))
    model.add(Dense(2, activation='sigmoid'))
    print(model.summary())'''

    '''model.add(Bidirectional(LSTM(20, return_sequences=True), input_shape=(train_X.shape[1], train_X.shape[2])))
    model.add(TimeDistributed(Dense(1, activation='sigmoid')))'''

    # model.compile(loss='mae', optimizer='adam')
    # it should be categorical
    # model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    print(model.summary())
    history = model.fit(train_X, train_y, epochs=100, batch_size=1, verbose=2)
    return model


#####################################################################
# serialize model to JSON
def savemodel(model):
    model_json = model.to_json()
    with open(myModelpath + itemname + "_model.json", "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    model.save_weights(myModelpath + itemname + "_model.h5")
    print("Saved model to disk")


#####################################################################
#####################################################################
df = read_csv(settingsPath + 'List.csv', header=None)
for index in range (df.shape[0]):
    itemname=df.iloc[index,0]
    print ('Current item :', itemname)

    #itemname = 'CSCO'

    dframe = getData(itemname)
    dframe = cleanData(dframe)
    djiframe = getData('^DJI')
    djiframe = cleanData(djiframe)
    # dframe=dframe.join(djiframe, lsuffix='_left', rsuffix='_right')
    dframe = dframe.join(djiframe, rsuffix='_right')

    fullFrame = engFeatures(dframe)
    reframed = rescale(fullFrame)

    train = reframed.values
    # split into input and outputs
    train_X, train_y = train[:, :-1], to_categorical(train[:, -1])
    train_X = numpy.reshape(train_X, (train_X.shape[0], train_X.shape[1], 1))
    #print(train_X.shape, train_y.shape, test_X.shape, test_y.shape)

    # test on entire dataframe
    test = reframed.values
    test_X, test_y = test[:, :-1], to_categorical(test[:, -1])
    test_X = numpy.reshape(test_X, (test_X.shape[0], test_X.shape[1], 1))

    themodel = buildmodel()
    #history = themodel.fit(train_X, train_y, epochs=100, batch_size=1, verbose=2)
    savemodel(themodel)




