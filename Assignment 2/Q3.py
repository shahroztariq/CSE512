import keras
import pandas
from keras import Sequential
from keras.layers import Dense
from sklearn import model_selection as sk
from keras.utils import np_utils
from sklearn import preprocessing
from keras.callbacks import LambdaCallback
from sklearn.metrics import classification_report
import seaborn as sns; sns.set(font_scale=1.2)
import numpy as np
from matplotlib import pyplot
from sklearn import svm
output_dim=3
nb_classes=3
input_dim=4
batch_size=80
nb_epoch=300
dataframe1 = pandas.read_csv("datasethw2/data1.txt",delimiter='\t', header=None,)
dataframe2 = pandas.read_csv("datasethw2/data2.txt",delimiter='\t', header=None,)
dataframe3 = pandas.read_csv("datasethw2/data3.txt",delimiter='\t', header=None,)
dataset=pandas.concat([dataframe1,dataframe2,dataframe3]).values
list1 = [1,2,3,4,5]
input_dim=len(list1)
X = np.array([[each_list[i] for i in list1] for each_list in dataset])
Y=(dataset[:,-1]-1)
X=X.astype('float')
X = preprocessing.normalize(X)
x_train, x_test, y_train, y_test = sk.train_test_split(X, Y, test_size=0.20, random_state=42)
clf=svm.SVC(gamma=0.01,C=100.,decision_function_shape='ovo')
clf.fit(x_train,y_train)
y_pred=clf.predict(x_test)
print(classification_report(y_test,y_pred))

# y_test=np.argmax(y_test,axis=1)
# y_pred=model.predict_classes(x_test)
# print(classification_report(y_test,y_pred))