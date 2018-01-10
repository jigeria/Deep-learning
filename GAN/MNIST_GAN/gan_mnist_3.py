import pickle as pkl
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data')


def model_inputs(real_dim, z_dim):
    """
    G와 D에 넣을 입력값인 inputs_real과 inputs_z를 리턴하는 함수

    Arguments
    ---------
    real_dim: 실제 인풋의 형태
    z_dim: 랜덤벡터 Z의 형태

    Returns
    -------
    inputs_real: D에 넣을 입력값
    inputs_z: G에 넣을 입력값
    """
    inputs_real = tf.placeholder(tf.float32, shape=(None, real_dim), name='input_real')
    inputs_z = tf.placeholder(tf.float32, shape=(None, z_dim), name='input_z')

    return inputs_real, inputs_z


def generator(z, out_dim, n_units=128, reuse=False, alpha=0.01):
    '''
    Generator 네트워크를 만든다.

    Arguments
    ---------
    z : Generator에 넣을 입력값(텐서)
    out_dim : 출력될 결과물의 형태
    n_units : 은닉층 유닛 갯수
    reuse : 재사용 여부
    alpha : Leaky ReLU에 넣을 leak 파라미터

    Returns
    -------
    out: 생성 결과
    '''
    with tf.variable_scope('generator', reuse=reuse):
        # Hidden layer
        h1 = tf.layers.dense(inputs=z, units=n_units, activation=None)

        # Leaky ReLU
        ## tf에 별도의 함수가 없어 이렇게 구현해야 한다.
        h1 = tf.maximum(h1 * alpha, h1)

        # Logits and tanh output
        logits = tf.layers.dense(inputs=h1, units=out_dim, activation=None)
        out = tf.tanh(logits)

        return out


def discriminator(x, n_units=128, reuse=False, alpha=0.01):
    '''
    Discriminator 네트워크를 만든다.

    Arguments
    ---------
    x : Discriminator에 넣을 입력값
    n_units : 은닉층 유닛 갯수
    reuse : 재사용 여부
    alpha : Leaky ReLU에 넣을 leak 파라미터

    Returns
    -------
    out: 분류 결과
    logits: sigmoid 직전 logits
    '''
    with tf.variable_scope('discriminator', reuse=reuse):
        # Hidden layer
        h1 = tf.layers.dense(x, n_units, activation=None)

        # Leaky ReLU
        h1 = tf.maximum(alpha * h1, h1)

        logits = tf.layers.dense(h1, 1, activation=None)
        out = tf.sigmoid(logits)  ## sigmoid를 쓴다.

        return out, logits

# D에 넣을 입력 데이터의 크기 (MNIST는 28*28인데 784개로 구성된 벡터로 변환해 넣으므로)
input_size = 784
# 랜덤벡터Z의 크기
z_size = 100
# G와 D의 은닉층 유닛 개수
g_hidden_size = 128
d_hidden_size = 128
# Leaky ReLU에 넣을 Leak 파라미터
alpha = 0.01
# 레이블 스무딩 파라미터
smooth = 0.1

tf.reset_default_graph()
# 입력값을 정의.
input_real, input_z = model_inputs(input_size, z_size)

# Generator 모델을 구현한다.
g_model = generator(input_z, input_size, n_units=g_hidden_size, alpha=alpha)

# Discriminator 모델을 구현한다.
## 여기서는 진짜와 가짜 2개를 만든다.
## 진짜와 가짜는 모두 같은 네트워크 가중치를 사용해야 하므로, reuse=True를 인자로 넘긴다.
d_model_real, d_logits_real = discriminator(input_real, n_units=d_hidden_size, alpha=alpha)
d_model_fake, d_logits_fake = discriminator(g_model, reuse=True, n_units=d_hidden_size, alpha=alpha)

# Calculate losses
## d_logits_real과 1의 차이가 진짜 데이터의 loss가 된다.
d_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=d_logits_real, labels=tf.ones_like(d_logits_real) * (1 - smooth)))

## d_logits_fake와 0의 차이가 가짜 데이터의 loss가 된다.
d_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=d_logits_fake, labels=tf.zeros_like(d_logits_real)))

## d_loss는 진짜와 가짜 loss의 합이다.
d_loss = d_loss_real + d_loss_fake

## g_loss는 가짜 loss와 1(가짜지만, Discriminator와 반대이므로 1)의 차이로 구한다.
g_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=d_logits_fake, labels=tf.ones_like(d_logits_fake)))

# Optimizers
learning_rate = 0.002

# 전체 변수를 가져온 다음, 이름으로 G와 D에 들어갈 변수를 추린다.
t_vars = tf.trainable_variables()
g_vars = [var for var in t_vars if var.name.startswith('generator')]
d_vars = [var for var in t_vars if var.name.startswith('discriminator')]

d_train_opt = tf.train.AdamOptimizer(learning_rate).minimize(d_loss, var_list=d_vars)
g_train_opt = tf.train.AdamOptimizer(learning_rate).minimize(g_loss, var_list=g_vars)