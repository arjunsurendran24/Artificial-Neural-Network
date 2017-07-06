#Importing Libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from keras.utils import np_utils


#Importing dataset
dataset = pd.read_csv('train.csv')

#Splitting inputs and outputs
X = dataset.iloc[:,1:785].values.astype(float)
y = dataset.iloc[:,0].values

#Encoding dependent variable
from sklearn.preprocessing import LabelEncoder
encoder = LabelEncoder()
encoder.fit(y)
y = encoder.transform(y)
# convert integers to dummy variables (i.e. one hot encoded)
y = np_utils.to_categorical(y)


# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

#Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

#Importing Libraries for ANN
import keras
from keras.models import Sequential
from keras.layers import Dense


#Initialisng the ANN
digitClassifier = Sequential()

#Input Layer and First Hidden Layer
digitClassifier.add(Dense(output_dim = 1024, activation = 'relu', input_dim = 784))

#Adding the Second hidden layer
digitClassifier.add(Dense(output_dim = 1024, activation = 'relu'))

# Adding the output layer
digitClassifier.add(Dense(output_dim = 10, activation = 'softmax'))

#Compiling the ANN
digitClassifier.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])

#Fitting ANN to the training set
digitClassifier.fit(X_train, y_train, batch_size = 10, epochs = 5)

# Predicting the Test set results
y_pred = digitClassifier.predict(X_test)
y_pred = (y_pred > 0.5)

#Inverse Transform of One Hot Encoding to calculate Confusion Matrix
y_prediction = y_pred.argmax(1)
y_test = y_test.argmax(1)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_prediction)

#Loading Test data
submission_set = pd.read_csv('test.csv')

#Creation of ImageId
id_ = [i for i in xrange(28001)]
id_ = id_[1:]
test_id = np.asarray(id_)

#Preparing Dataset
X_submission = submission_set.values.astype(float)
X_submission = sc.transform(X_submission)

#Predicting the Result
y = digitClassifier.predict(X_submission)
y  = (y > 0.5)
y = y.argmax(1)

#Writing to CSV
CSVFile = np.column_stack((test_id.T,y.T))
outdata = pd.DataFrame(CSVFile, columns = ['ImageId', 'Label'])
outdata.to_csv('output.csv', index = None)
