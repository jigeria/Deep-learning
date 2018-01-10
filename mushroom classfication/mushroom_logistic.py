from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
labelencoder=LabelEncoder()
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
nb_classes = 2

xy = pd.read_csv('mushrooms.csv')

a = xy.isnull().sum()

for col in xy.columns:
    xy[col] = labelencoder.fit_transform(xy[col])

print(xy)

array = xy.values.tolist()

y_data = []

for i in range(0, 8124):
    y_data.append([array[i][0]])

for i in range(0, len(array[0])):
    for j in range(0, len(array)):
        array[j][i] = float(array[j][i])

for row in array: # cap_color / gill_size / gill_color
    del row[0]
    del row[3]
    del row[18]
    del row[18]

'''
for row in array: # cap_color / gill_size / gill_color
    del row[0]
    del row[0]
    del row[0]
    del row[1]
    del row[1]
    del row[1]
    del row[1]
    del row[3]
    del row[3]
    del row[3]
    del row[3]
    del row[3]
    del row[3]
    del row[3]
    del row[3]
    del row[3]
    del row[3]
    del row[3]
    del row[3]
    del row[3]

'''

std_data = StandardScaler().fit_transform(array)

x_data = std_data

print(x_data)
print(y_data)

X = tf.placeholder(tf.float32, shape=[None, 19])
Y = tf.placeholder(tf.float32, shape=[None, 1])

W1 = tf.get_variable("W1", shape=[19, 128], initializer=tf.contrib.layers.xavier_initializer())
b1 = tf.Variable(tf.random_normal([128]))
L1 = tf.nn.relu(tf.matmul(X, W1) + b1)

W2 = tf.get_variable("W2", shape=[128, 128], initializer=tf.contrib.layers.xavier_initializer())
b2 = tf.Variable(tf.random_normal([128]))
L2 = tf.nn.relu(tf.matmul(L1, W2) + b2)

W3 = tf.get_variable('W3', shape=[128, 1], initializer=tf.contrib.layers.xavier_initializer())
b3 = tf.Variable(tf.random_normal([1]))

hypothesis = tf.sigmoid(tf.matmul(L2, W3) + b3)

cost = -tf.reduce_mean(Y * tf.log(hypothesis) + (1 - Y) * tf.log(1-hypothesis))
train = tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(cost)

predicted = tf.cast(hypothesis > 0.5, dtype=tf.float32)
accuracy = tf.reduce_mean(tf.cast(tf.equal(predicted, Y), dtype=tf.float32))

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for step in range(5001):
        cost_val, _ = sess.run([cost, train], feed_dict={X: x_data, Y: y_data})
        if step % 200 == 0:
            print(step, cost_val)

    h, c, a = sess.run([hypothesis, predicted, accuracy],feed_dict={X: x_data, Y: y_data})
    print("\nHypothesis: ", h, "\nCorrect (Y): ", c, "\nAccuracy: ", a)

