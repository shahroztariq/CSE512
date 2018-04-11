import tensorflow as tf
import pandas
import numpy
from keras.optimizers import RMSprop
from keras import Sequential, optimizers
from sklearn import model_selection as sk
from keras.layers import Dense, Dropout
from sklearn.preprocessing import StandardScaler
import math
from keras.callbacks import LambdaCallback
from matplotlib import pyplot
train_percent=0.75
batch_size=20
epochs=100
dataframe = pandas.read_csv("bitcoin\export-EtherPrice.csv")
dataset=dataframe.values
X=[math.log(row[1]) for row in dataset]
Y=[row[2] for row in dataset]

#Y=[math.log(row[2]) for row in dataset]
s = sum(Y)
Y = [float(i)/s for i in Y]
s=max(Y)
Y = [float(i)/s for i in Y]
X=numpy.array(X[:-1])
Y=numpy.array(Y[:-1])
X = X.reshape(-1, 1).astype('float')
Y = Y.reshape(-1, 1).astype('float')
sc=StandardScaler()
X=sc.fit_transform(X)
#Y=sc.fit_transform(Y)
print(X)
print(Y)
x_train, x_test, y_train, y_test = sk.train_test_split(X, Y, test_size=0.10, random_state=42,shuffle=False)
print(x_train.shape[0], 'train samples')
print_weights = LambdaCallback(on_epoch_end=lambda batch, logs: print("Weight & Bias:"+str(model.layers[0].get_weights())))
model=Sequential()
model.add(Dense(1,input_shape=(1,),activation='linear'))
print(model.summary())
sgd = optimizers.SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='mse',
              optimizer=sgd,
              metrics=['mse','accuracy'])
history = model.fit(x_train, y_train,
                    batch_size=batch_size,
                    epochs=epochs,
                    verbose=2,
                    validation_data=(x_test, y_test),
                    callbacks=[print_weights])
score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print(score)
#X1=sc.fit_transform(1521417600)
#print(model.predict_classes(X1,verbose=1))

pyplot.figure('MSE')
pyplot.plot(history.history['mean_squared_error'])
pyplot.figure('Acc')
pyplot.plot(history.history['acc'])
pyplot.show()
