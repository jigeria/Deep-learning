import os
os.environ['KERAS_BACKEND']='tensorflow' # can choose theano, tensorflow, cntk
os.environ['THEANO_FLAGS']='floatX=float32,device=cuda,optimizer=fast_run,dnn.library_path=/usr/lib'

import keras.backend as K
K.set_image_data_format('channels_last')
channel_axis = -1

from keras.models import Sequential, Model
from keras.layers import Conv2D, ZeroPadding2D, BatchNormalization, Input, Dropout
from keras.layers import Conv2DTranspose, Reshape, Activation, Cropping2D, Flatten
from keras.layers import Concatenate, Dense, merge, Lambda
from keras.layers.advanced_activations import LeakyReLU
from keras.activations import relu
from keras.initializers import RandomNormal
from IPython.display import SVG
from keras.utils.vis_utils import model_to_dot
from keras.optimizers import RMSprop, SGD, Adam
from PIL import Image
import numpy as np
import glob
from random import randint, shuffle


# Weights initializations
# bias are initailized as 0
def __conv_init(a):
    print("conv_init", a)
    k = RandomNormal(0, 0.02)(a) # for convolution kernel
    k.conv_weight = True
    return k
conv_init = RandomNormal(0, 0.02)
gamma_init = RandomNormal(1., 0.02) # for batch normalization


def conv2d(f, *a, **k):
    return Conv2D(f, kernel_initializer=conv_init, *a, **k)


def batchnorm():
    return BatchNormalization(momentum=0.9, axis=channel_axis, epsilon=1.01e-5,
                              gamma_initializer=gamma_init)


def num_patches(output_img_dim=(256, 256, 3), sub_patch_dim=(64, 64)):
    """
    Creates non-overlaping patches to feed to the PATCH GAN
    (Section 2.2.2 in paper)
    The paper provides 3 options.
    Pixel GAN = 1x1 patches (aka each pixel)
    PatchGAN = nxn patches (non-overlaping blocks of the image)
    ImageGAN = im_size x im_size (full image)
    Ex: 4x4 image with patch_size of 2 means 4 non-overlaping patches
    :param output_img_dim:
    :param sub_patch_dim:
    :return:
    """

    # num of non-overlaping patches
    nb_non_overlaping_patches = (output_img_dim[0] / sub_patch_dim[0]) * (output_img_dim[1] / sub_patch_dim[1])

    # dimensions for the patch discriminator
    patch_disc_img_dim = (sub_patch_dim[0], sub_patch_dim[1], output_img_dim[2])

    return int(nb_non_overlaping_patches), patch_disc_img_dim
'''
def BASIC_D(nc_in, nc_out, ndf, max_layers=3):
    """DCGAN_D(nc, ndf, max_layers=3)
       nc: channels
       ndf: filters of the first layer
       max_layers: max hidden layers
    """
    input_a, input_b =  Input(shape=(None, None, nc_in)), Input(shape=(None, None, nc_out))

    _ = Concatenate(axis=channel_axis)([input_a, input_b])
    _ = conv2d(ndf, kernel_size=4, strides=2, padding="same", name='First')(_)
    _ = LeakyReLU(alpha=0.2)(_)

    for layer in range(1, max_layers):
        out_feat = ndf * min(2 ** layer, 8)
        _ = conv2d(out_feat, kernel_size=4, strides=2, padding="same",
                   use_bias=False, name='pyramid.{0}'.format(layer)
                   )(_)
        _ = batchnorm()(_, training=1)
        _ = LeakyReLU(alpha=0.2)(_)

    out_feat = ndf * min(2 ** max_layers, 8)
    _ = ZeroPadding2D(1)(_)
    _ = conv2d(out_feat, kernel_size=4, use_bias=False, name='pyramid_last')(_)
    _ = batchnorm()(_, training=1)
    _ = LeakyReLU(alpha=0.2)(_)

    # final layer
    # _ = ZeroPadding2D(1)(_)
    #_ = conv2d(1, kernel_size=4, name='final'.format(out_feat, 1),activation="sigmoid")(_)

    im_width = im_height = 256
    output_channels = 1
    output_img_dim = (im_width, im_height, output_channels)

    sub_patch_dim = (256, 256)
    nb_patch_patches, patch_gan_dim = num_patches(output_img_dim=output_img_dim, sub_patch_dim=sub_patch_dim)

    patch_gan_discriminator = generate_patch_gan_loss(last_disc_conv_layer=_,
                                                      patch_dim=patch_gan_dim,
                                                      input_a=input_a,
                                                      input_b=input_b,
                                                      nb_patches=nb_patch_patches,
                                                      out_feat=out_feat)
    #return patch_gan_discriminator

    return Model(inputs=[input_a, input_b], outputs=_)
'''

def BASIC_D(nc_in, nc_out, ndf, max_layers=3):
    """DCGAN_D(nc, ndf, max_layers=3)
       nc: channels
       ndf: filters of the first layer
       max_layers: max hidden layers
    """
    input_a, input_b = Input(shape=(None, None, nc_in)), Input(shape=(None, None, nc_out))

    _ = Concatenate(axis=channel_axis)([input_a, input_b])
    _ = conv2d(ndf, kernel_size=4, strides=2, padding="same", name='First')(_)
    _ = LeakyReLU(alpha=0.2)(_)

    for layer in range(1, max_layers):
        out_feat = ndf * min(2 ** layer, 8)
        _ = conv2d(out_feat, kernel_size=4, strides=2, padding="same",
                   use_bias=False, name='pyramid.{0}'.format(layer)
                   )(_)
        _ = batchnorm()(_, training=1)
        _ = LeakyReLU(alpha=0.2)(_)

    out_feat = ndf * min(2 ** max_layers, 8)
    _ = ZeroPadding2D(1)(_)
    _ = conv2d(out_feat, kernel_size=4, use_bias=False, name='pyramid_last')(_)
    _ = batchnorm()(_, training=1)
    _ = LeakyReLU(alpha=0.2)(_)

    # final layer
    _ = ZeroPadding2D(1)(_)
    _ = conv2d(1, kernel_size=4, name='final'.format(out_feat, 1),
               activation="sigmoid")(_)

    im_width = im_height = 256
    output_channels = 1
    output_img_dim = (im_width, im_height, output_channels)

    sub_patch_dim = (256, 256)
    nb_patches, patch_dim = num_patches(output_img_dim=output_img_dim, sub_patch_dim=sub_patch_dim)

    input_layer = Concatenate(axis=channel_axis)([input_a, input_b])

    # generate a list of inputs for the different patches to the network
    list_input = [Input(shape=patch_dim, name="patch_gan_input_%s" % i) for i in range(nb_patches)]

    # get an activation
    x_flat = Flatten()(_)
    x = Dense(2, activation='softmax', name="disc_dense")(x_flat)

    patch_gan = Model(input=[input_layer], output=[x, x_flat], name="patch_gan")

    # generate individual losses for each patch
    x = [patch_gan(patch)[0] for patch in list_input]
    x_mbd = [patch_gan(patch)[1] for patch in list_input]

    # merge layers if have multiple patches (aka perceptual loss)
    if len(x) > 1:
        x = merge(x, mode="concat", name="merged_features")
    else:
        x = x[0]

    # merge mbd if needed
    # mbd = mini batch discrimination
    # https://arxiv.org/pdf/1606.03498.pdf
    if len(x_mbd) > 1:
        x_mbd = merge(x_mbd, mode="concat", name="merged_feature_mbd")
    else:
        x_mbd = x_mbd[0]

    num_kernels = 100
    dim_per_kernel = 5

    M = Dense(num_kernels * dim_per_kernel, bias=False, activation=None)
    MBD = Lambda(minb_disc, output_shape=lambda_output)

    x_mbd = M(x_mbd)
    x_mbd = Reshape((num_kernels, dim_per_kernel))(x_mbd)
    x_mbd = MBD(x_mbd)
    x = merge([x, x_mbd], mode='concat')

    x_out = Dense(2, activation="softmax", name="disc_output")(x)

    discriminator = Model(input=list_input, output=[x_out], name='discriminator_nn')

    return discriminator

    '''
    patch_gan_discriminator = generate_patch_gan_loss(last_disc_conv_layer=_,
                                                      patch_dim=patch_gan_dim,
                                                      input_a=input_a,
                                                      input_b=input_b,
                                                      nb_patches=nb_patch_patches)
    
    return patch_gan_discriminator
    '''
    #return Model(inputs=[input_a, input_b], outputs=_)


'''
    patch_gan_discriminator = generate_patch_gan_loss(last_disc_conv_layer=_,
                                                      patch_dim=patch_gan_dim,
                                                      input_a=input_a,
                                                      input_b=input_b,
                                                      nb_patches=nb_patch_patches,
                                                      out_feat=out_feat)
    # return patch_gan_discriminator
'''



def generate_patch_gan_loss(last_disc_conv_layer, patch_dim, input_a, input_b, nb_patches):

    input_layer = Concatenate(axis=channel_axis)([input_a, input_b])

    # generate a list of inputs for the different patches to the network
    list_input = [Input(shape=patch_dim, name="patch_gan_input_%s" % i) for i in range(nb_patches)]

    # get an activation
    x_flat = Flatten()(last_disc_conv_layer)
    x = Dense(2, activation='softmax', name="disc_dense")(x_flat)

    patch_gan = Model(input=[input_layer], output=[x, x_flat], name="patch_gan")

    # generate individual losses for each patch
    x = [patch_gan(patch)[0] for patch in list_input]
    x_mbd = [patch_gan(patch)[1] for patch in list_input]

    # merge layers if have multiple patches (aka perceptual loss)
    if len(x) > 1:
        x = merge(x, mode="concat", name="merged_features")
    else:
        x = x[0]

    # merge mbd if needed
    # mbd = mini batch discrimination
    # https://arxiv.org/pdf/1606.03498.pdf
    if len(x_mbd) > 1:
        x_mbd = merge(x_mbd, mode="concat", name="merged_feature_mbd")
    else:
        x_mbd = x_mbd[0]

    num_kernels = 100
    dim_per_kernel = 5

    M = Dense(num_kernels * dim_per_kernel, bias=False, activation=None)
    MBD = Lambda(minb_disc, output_shape=lambda_output)

    x_mbd = M(x_mbd)
    x_mbd = Reshape((num_kernels, dim_per_kernel))(x_mbd)
    x_mbd = MBD(x_mbd)
    x = merge([x, x_mbd], mode='concat')

    x_out = Dense(2, activation="softmax", name="disc_output")(x)

    discriminator = Model(input=list_input, output=[x_out], name='discriminator_nn')

    return discriminator

def lambda_output(input_shape):
    return input_shape[:2]


def minb_disc(x):
    diffs = K.expand_dims(x, 3) - K.expand_dims(K.permute_dimensions(x, [1, 2, 0]), 0)
    abs_diffs = K.sum(K.abs(diffs), 2)
    x = K.sum(K.exp(-abs_diffs), 2)

    return x



def UNET_G(isize, nc_in=3, nc_out=3, ngf=64, fixed_input_size=True):
    max_nf = 8 * ngf

    def block(x, s, nf_in, use_batchnorm=True, nf_out=None, nf_next=None):
        # print("block",x,s,nf_in, use_batchnorm, nf_out, nf_next)
        assert s >= 2 and s % 2 == 0
        if nf_next is None:
            nf_next = min(nf_in * 2, max_nf)
        if nf_out is None:
            nf_out = nf_in
        x = conv2d(nf_next, kernel_size=4, strides=2, use_bias=(not (use_batchnorm and s > 2)),
                   padding="same", name='conv_{0}'.format(s))(x)
        if s > 2:
            if use_batchnorm:
                x = batchnorm()(x, training=1)
            x2 = LeakyReLU(alpha=0.2)(x)
            x2 = block(x2, s // 2, nf_next)
            x = Concatenate(axis=channel_axis)([x, x2])
        x = Activation("relu")(x)
        x = Conv2DTranspose(nf_out, kernel_size=4, strides=2, use_bias=not use_batchnorm,
                            kernel_initializer=conv_init,
                            name='convt.{0}'.format(s))(x)
        x = Cropping2D(1)(x)
        if use_batchnorm:
            x = batchnorm()(x, training=1)
        if s <= 8:
            x = Dropout(0.5)(x, training=1)
        return x

    s = isize if fixed_input_size else None

    _ = inputs = Input(shape=(s, s, nc_in))
    _ = block(_, isize, nc_in, False, nf_out=nc_out, nf_next=ngf)
    _ = Activation('tanh')(_)
    return Model(inputs=inputs, outputs=[_])

nc_in = 3
nc_out = 3
ngf = 64
ndf = 64
lamda = 10

loadSize = 286
imageSize = 256
batchSize = 1
lrD = 2e-4
lrG = 2e-4

netD = BASIC_D(nc_in, nc_out, ndf)
netD.summary()

netG = UNET_G(imageSize, nc_in, nc_out, ngf)
SVG(model_to_dot(netG, show_shapes=True).create(prog='dot', format='svg'))
netG.summary()

real_A = netG.input
fake_B = netG.output
netG_generate = K.function([real_A], [fake_B])
real_B = netD.inputs[1]
output_D_real = netD([real_A, real_B]) #coditional GAN
output_D_fake = netD([real_A, fake_B]) #coditional GAN

loss_fn = lambda output, target : -K.mean(K.log(output+1e-12)*target+K.log(1-output+1e-12)*(1-target))

loss_D_real = loss_fn(output_D_real, K.ones_like(output_D_real))
loss_D_fake = loss_fn(output_D_fake, K.zeros_like(output_D_fake))
loss_G_fake = loss_fn(output_D_fake, K.ones_like(output_D_fake))

loss_L1 = K.mean(K.abs(fake_B-real_B))

loss_D = loss_D_real + loss_D_fake
training_updates = Adam(lr=lrD, beta_1=0.5).get_updates(netD.trainable_weights, [], loss_D)
netD_train = K.function([real_A, real_B], [loss_D / 2.0], training_updates)
loss_G = loss_G_fake + 100 * loss_L1
training_updates = Adam(lr=lrG, beta_1=0.5).get_updates(netG.trainable_weights, [], loss_G)
netG_train = K.function([real_A, real_B], [loss_G_fake, loss_L1], training_updates)

def load_data(file_pattern):
    return glob.glob(file_pattern)

def read_image(fn, direction=0):
    im = Image.open(fn)
    im = im.resize((loadSize * 2, loadSize), Image.BILINEAR)
    arr = np.array(im) / 255.0 * 2 - 1
    w1, w2 = (loadSize - imageSize) // 2, (loadSize + imageSize) // 2
    h1, h2 = w1, w2

    imgA = arr[h1:h2, loadSize + w1:loadSize + w2, :]
    imgB = arr[h1:h2, w1:w2, :]

    if direction == 0:
        return imgA, imgB
    else:
        return imgB, imgA

direction = 1

trainAB = load_data('./cards_ab/cards_ab/train/*.jpg')
valAB = load_data('./cards_ab/cards_ab/val/*.jpg')

assert len(trainAB) and len(valAB)

def minibatch(dataAB, batchsize, direction=0):
    length = len(dataAB)
    epoch = i = 0
    tmpsize = None
    while True:
        size = tmpsize if tmpsize else batchsize
        if i + size > length:
            shuffle(dataAB)
            i = 0
            epoch += 1
        dataA = []
        dataB = []
        for j in range(i, i + size):
            imgA, imgB = read_image(dataAB[j], direction)
            dataA.append(imgA)
            dataB.append(imgB)
        dataA = np.float32(dataA)
        dataB = np.float32(dataB)
        i += size
        tmpsize = yield epoch, dataA, dataB

from IPython.display import display

def showX(X, rows=1):
    assert X.shape[0] % rows == 0

    int_X = ((X + 1) / 2.0 * 255).clip(0, 255).astype('uint8')
    int_X = int_X.reshape(-1, imageSize, imageSize, 3)
    int_X = int_X.reshape(rows, -1, imageSize, imageSize, 3).swapaxes(1, 2).reshape(rows * imageSize, -1, 3)

    display(Image.fromarray(int_X))


train_batch = minibatch(trainAB, 6, direction=direction)
_, trainA, trainB = next(train_batch)
showX(trainA)
showX(trainB)
del train_batch, trainA, trainB

def netG_gen(A):
    return np.concatenate([netG_generate([A[i:i + 1]])[0] for i in range(A.shape[0])], axis=0)

import time
from IPython.display import clear_output

t0 = time.time()
niter = 50
gen_iterations = 0
errL1 = epoch = errG = 0
errL1_sum = errG_sum = errD_sum = 0

display_iters = 500
val_batch = minibatch(valAB, 6, direction)
train_batch = minibatch(trainAB, batchSize, direction)

while epoch < niter:
    epoch, trainA, trainB = next(train_batch)
    errD, = netD_train([trainA, trainB])
    errD_sum += errD

    errG, errL1 = netG_train([trainA, trainB])
    errG_sum += errG
    errL1_sum += errL1
    gen_iterations += 1
    if gen_iterations % display_iters == 0:
        if gen_iterations % (5 * display_iters) == 0:
            clear_output()
        print('[%d/%d][%d] Loss_D: %f Loss_G: %f loss_L1: %f'
              % (epoch, niter, gen_iterations, errD_sum / display_iters, errG_sum / display_iters,
                 errL1_sum / display_iters), time.time() - t0)
        _, valA, valB = train_batch.send(6)
        fakeB = netG_gen(valA)
        showX(np.concatenate([valA, valB, fakeB], axis=0), 3)
        errL1_sum = errG_sum = errD_sum = 0
        _, valA, valB = next(val_batch)
        fakeB = netG_gen(valA)
        showX(np.concatenate([valA, valB, fakeB], axis=0), 3)
_, valA, valB = val_batch.send(6)
fakeB = netG_gen(valA)
showX(np.concatenate([valA, valB, fakeB], axis=0), 3)
errL1_sum = errG_sum = errD_sum = 0
_, valA, valB = next(val_batch)
fakeB = netG_gen(valA)
showX(np.concatenate([valA, valB, fakeB], axis=0), 3)


def read_single_image(fn):
    im = Image.open(fn)
    im = im.resize((loadSize, loadSize), Image.BILINEAR)
    arr = np.array(im) / 255.0 * 2 - 1
    w1, w2 = (loadSize - imageSize) // 2, (loadSize + imageSize) // 2
    h1, h2 = w1, w2
    img = arr[h1:h2, w1:w2, :]

    return img


# 0~636

max_idx = 636
src_dir = './cards_ab/cards_ab/test_in/'
dst_dir = './cards_ab/cards_ab/test_out/'

for idx in range(max_idx + 1):
    data = []

    src_path = src_dir + str(idx) + '.jpg'
    dst_path = dst_dir + str(idx) + '.jpg'

    src_img = read_single_image(src_path)
    data.append(src_img)
    data = np.float32(data)

    fake = netG_gen(data)

    int_X = ((fake[0] + 1) / 2.0 * 255).clip(0, 255).astype('uint8')
    int_X = int_X.reshape(-1, imageSize, imageSize, 3)
    int_X = int_X.reshape(1, -1, imageSize, imageSize, 3).swapaxes(1, 2).reshape(1 * imageSize, -1, 3)

    dst_img = Image.fromarray(int_X)
    dst_img.save(dst_path, quality=90)