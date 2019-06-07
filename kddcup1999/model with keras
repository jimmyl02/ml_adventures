import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt

from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.wrappers.scikit_learn import KerasClassifier
from keras.utils import np_utils
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline

# fix random seed for reproducibility
seed = 7
np.random.seed(seed)

Data = pd.read_csv('C:\projects\kddcupBinaryFull.data.csv', header=None, skiprows=1)
DataValues = Data.values

X = DataValues[:, 0:40]
Y = DataValues[:, 41]

#Do all of the one hot encoding
#Encoding for the labels
encoder = LabelEncoder()
encoder.fit(Y)
encoded_Y = encoder.transform(Y)

dummy_Y = np_utils.to_categorical(encoded_Y)

#Encoding for the data
encoder_X_1 = LabelEncoder()
X[:, 1] = encoder_X_1.fit_transform(X[:, 1])
encoder_X_2 = LabelEncoder()
X[:, 2] = encoder_X_2.fit_transform(X[:, 2])
encoder_X_3 = LabelEncoder()
X[:, 3] = encoder_X_3.fit_transform(X[:, 3])

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.15, random_state=42)

#print(y_train)

def shallowModel():
#initialize model
model = Sequential()
model.add(Dense(80, input_dim=40, activation="relu"))
model.add(Dense(160, activation="relu"))
model.add(Dropout(0.2))
model.add(Dense(160, activation="relu"))
model.add(Dropout(0.2))
model.add(Dense(160, activation="relu"))
model.add(Dropout(0.2))
model.add(Dense(80, activation="relu"))
model.add(Dropout(0.2))
model.add(Dense(40, activation="relu"))
model.add(Dense(20, activation="softmax"))
#compile the model
model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=['accuracy'])
return model
clf = KerasClassifier(build_fn=shallowModel, epochs=200, batch_size=16, verbose=2)

kfold = KFold(n_splits=10, shuffle=True, random_state=seed)

results = cross_val_score(clf, X, dummy_Y, cv=kfold)

print("Baseline: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))
