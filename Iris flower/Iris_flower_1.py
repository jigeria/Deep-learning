import tensorflow as tf
import numpy as np
import pandas as pd

tf.set_random_seed(777)

xy = pd.read_csv('Iris.csv')

array = xy.values.tolist()
y_data = []
# y_data.append()

for i in range(0, 150):
    y_data.append([array[i][5]])

for row in array:
    del row[0]
    del row[4]

x_data = array

nb_classes = 3

for i in range(0, 150):
    if 'Iris-setosa' in y_data[i]:
        y_data[i] = [0]
    elif 'Iris-versicolor' in y_data[i]:
        y_data[i] = [1]
    elif 'Iris-virginica' in y_data[i]:
        y_data[i] = [2]

X = tf.placeholder(tf.float32, [None, 4])
Y = tf.placeholder(tf.int32, [None, 1])

Y_one_hot = tf.one_hot(Y, nb_classes)
print("one hot", Y_one_hot)
Y_one_hot = tf.reshape(Y_one_hot, [-1, nb_classes])
print("one hot", Y_one_hot)

W = tf.Variable(tf.random_normal([4, nb_classes]), name='weight')
b = tf.Variable(tf.random_normal([nb_classes]), name='bias')

logits = tf.matmul(X, W) + b
hypothesis = tf.nn.softmax(logits)

cost_i = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=Y_one_hot)
cost = tf.reduce_mean(cost_i)
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.1).minimize(cost)

prediction = tf.argmax(hypothesis, 1)
correct_prediction = tf.equal(prediction, tf.argmax(Y_one_hot, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for step in range(20001):
        sess.run(optimizer, feed_dict={X: x_data, Y: y_data})
        if step % 100 == 0:
            loss, acc = sess.run([cost, accuracy], feed_dict={X: x_data, Y: y_data})
            print("Step: {:5}\tLoss: {:.3f}\tAcc: {:.2%}".format(step, loss, acc))
