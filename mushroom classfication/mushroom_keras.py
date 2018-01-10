from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from keras.utils import np_utils
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Activation
labelencoder=LabelEncoder()
import pandas as pd
import matplotlib.pyplot as plt
nb_classes = 2

xy = pd.read_csv('mushrooms.csv')

a = xy.isnull().sum()

for col in xy.columns:
    xy[col] = labelencoder.fit_transform(xy[col])

array = xy.values.tolist()

y_data = []

for i in range(0, 8124):
    y_data.append([array[i][0]])

for i in range(0, len(array[0])):
    for j in range(0, len(array)):
        array[j][i] = float(array[j][i])

for row in array: # 19
    del row[0]
    del row[3]
    del row[18]
    del row[18]


std_data = StandardScaler().fit_transform(array)

x_data = std_data

print(x_data)
print(y_data)

x_train = x_data[:6000]
y_train = y_data[:6000]
x_test = x_data[6000:]
y_test = y_data[6000:]

#y_data = np_utils.to_categorical(y_data)

model = Sequential()
model.add((Dense(128, input_dim=19, activation='relu')))
model.add(Dense(128, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(1,  activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

model.fit(x_train, y_train, epochs=100, batch_size=32)

scores = model.evaluate(x_test,  y_test)
print("%s: %.2f%%" %(model.metrics_names[1], scores[1]*100))


#print(y_data)

