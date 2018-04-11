import keras
import pandas
from keras import Sequential
from keras.layers import Dense
from sklearn import model_selection as sk
from keras.utils import np_utils
from sklearn import preprocessing
from sklearn.metrics import classification_report
from keras.callbacks import LambdaCallback
from matplotlib import pyplot
import numpy as np
import seaborn as sns; sns.set(font_scale=1.2)
output_dim=3
nb_classes=3
input_dim=5
batch_size=80
nb_epoch=300
dataframe1 = pandas.read_csv("datasethw2/data1.txt",delimiter='\t', header=None,)
dataframe2 = pandas.read_csv("datasethw2/data2.txt",delimiter='\t', header=None,)
dataframe3 = pandas.read_csv("datasethw2/data3.txt",delimiter='\t', header=None,)
dataset=pandas.concat([dataframe1,dataframe2,dataframe3]).values
print(len(dataset))
X=dataset[:,:5]
Y=(dataset[:,-1]-1)
X=X.astype('float')
Y=Y.reshape(-1, 1)
X = preprocessing.normalize(X)
x_train, x_test, y_train, y_test = sk.train_test_split(X, Y, test_size=0.20, random_state=42)
y_train=np_utils.to_categorical(y_train,nb_classes)
y_test=np_utils.to_categorical(y_test,nb_classes)
print_weights = LambdaCallback(on_epoch_end=lambda batch, logs: print("Weights\n"+str(model.layers[0].get_weights()[0])+"\nBais\n"+str(model.layers[0].get_weights()[1])+"\n"))
model=Sequential()
model.add(Dense(3,input_dim=input_dim,activation='softmax'))
print(model.summary())
model.compile(optimizer=keras.optimizers.adam(),loss='categorical_crossentropy',metrics=['accuracy',keras.metrics.categorical_accuracy])
history=model.fit(x_train,y_train,
                  batch_size=batch_size,
                  nb_epoch=nb_epoch,
                  verbose=0,
                  validation_data=(x_test,y_test),
                  callbacks=[print_weights])
score=model.evaluate(x_test,y_test,verbose=0)
print('Test loss:',score[0])
print('Test accuracy:',score[1])
pyplot.figure('categorical_accuracy')
pyplot.xlabel('Iterations')
pyplot.ylabel('Categorical Accuracy')
pyplot.plot(history.history['categorical_accuracy'])
pyplot.show()
y_test=np.argmax(y_test,axis=1)
y_pred=model.predict_classes(x_test)
print(classification_report(y_test,y_pred))