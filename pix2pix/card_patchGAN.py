import os
os.environ['KERAS_BACKEND']='tensorflow' # can choose theano, tensorflow, cntk
os.environ['THEANO_FLAGS']='floatX=float32,device=cuda,optimizer=fast_run,dnn.library_path=/usr/lib'

import keras.backend as K
K.set_image_data_format('channels_last')
K.set_image_dim_ordering('tf')

channel_axis = -1

from keras.models import Sequential, Model
from keras.layers import Conv2D, ZeroPadding2D, BatchNormalization, Input, Dropout
from keras.layers import Conv2DTranspose, Reshape, Activation, Cropping2D, Flatten
from keras.layers import Concatenate, Dense, merge, Lambda, InputLayer
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


def generate_patch_gan_loss(last_disc_conv_layer, patch_dim, input_layer, nb_patches):
    # generate a list of inputs for the different patches to the network
    list_input_A = [Input(shape=patch_dim, name="patch_gan_input_%s" % i) for i in range(nb_patches)]
    list_input_B = [Input(shape=patch_dim, name="patch_gan_input_%s" % i) for i in range(nb_patches)]

    concat_list = []

    for a, b in zip(list_input_A, list_input_B):
        concat_list.append(Concatenate(axis=channel_axis)([a, b]))
        print(Concatenate(axis=channel_axis)([a, b]).shape)

    # get an activation
    x_flat = Flatten()(last_disc_conv_layer)
    x = Dense(2, activation='softmax', name="disc_dense")(x_flat)
    print(x.shape)
    patch_gan = Model(input=input_layer, output=[x, x_flat], name="patch_gan")

    # generate individual losses for each patch
    x = [patch_gan(patch)[0] for patch in concat_list]
    x_mbd = [patch_gan(patch)[1] for patch in concat_list]

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


    discriminator = Model(input=[list_input_A, list_input_B], output=[x_out], name='discriminator_nn')
    return discriminator


def lambda_output(input_shape):
    return input_shape[:2]

def minb_disc(x):
    diffs = K.expand_dims(x, 3) - K.expand_dims(K.permute_dimensions(x, [1, 2, 0]), 0)
    abs_diffs = K.sum(K.abs(diffs), 2)
    x = K.sum(K.exp(-abs_diffs), 2)

    return x


def BASIC_D(nc_in, nc_out, ndf, max_layers=3):
    """
    DCGAN_D(nc, ndf, max_layers=3)
    nc: channels
    ndf: filters of the first layer
    max_layers: max hidden layers
    """

    input_a = Input(shape=(64, 64, nc_in * 2))
    # 두개의 인풋을 concat
    # CNN Layer
    _ = conv2d(ndf, kernel_size=4, strides=2, padding="same", name='First')(input_a)
    _ = LeakyReLU(alpha=0.2)(_)
    # layer를 max_layer 만큼 중첩해서 쌓음
    for layer in range(1, max_layers):
        out_feat = ndf * min(2 ** layer, 8)
        _ = conv2d(out_feat, kernel_size=4, strides=2, padding="same",
                   use_bias=False, name='pyramid.{0}'.format(layer)
                   )(_)
        _ = batchnorm()(_, training=1)
        _ = LeakyReLU(alpha=0.2)(_)
        print(_)

    out_feat = ndf * min(2 ** max_layers, 8)
    _ = ZeroPadding2D(1)(_)
    _ = conv2d(out_feat, kernel_size=4, use_bias=False, name='pyramid_last')(_)
    _ = batchnorm()(_, training=1)
    _ = LeakyReLU(alpha=0.2)(_)
    par_layer = _
    print(_)


    nb_patch_patches, patch_gan_dim = num_patches(output_img_dim=(256, 256, 3), sub_patch_dim=(64, 64))

    print("nb_patch_patches, patch_gan_dim:" + str(nb_patch_patches), str(patch_gan_dim))

    patch_gan_discriminator = generate_patch_gan_loss(last_disc_conv_layer=par_layer,
                                                      patch_dim=patch_gan_dim,
                                                      input_layer=input_a,
                                                      nb_patches=nb_patch_patches)
    return patch_gan_discriminator



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

def extract_patches(images, sub_patch_dim):
    """
    Cuts images into k subpatche
    Each kth cut as the kth patches for all images
    ex: input 3 images [im1, im2, im3]
    output [[im_1_patch_1, im_2_patch_1], ... , [im_n-1_patch_k, im_n_patch_k]]
    :param images: array of Images (num_images, im_channels, im_height, im_width)
    :param sub_patch_dim: (height, width) ex: (30, 30) Subpatch dimensions
    :return:
    """
    im_height, im_width = images.shape[1:3]
    patch_height, patch_width = sub_patch_dim

    # list out all xs  ex: 0, 29, 58, ...
    x_spots = range(0, im_width, patch_width)

    # list out all ys ex: 0, 29, 58
    y_spots = range(0, im_height, patch_height)
    all_patches = []

    print("start patch")
    for y in y_spots:
        for x in x_spots:
            # indexing here is cra
            # images[num_images, num_channels, width, height]
            # this says, cut a patch across all images at the same time with this width, height
            image_patches = images[:, y: y+patch_height, x: x+patch_width, :]
            all_patches.append(np.asarray(image_patches, dtype=np.float32))

    #print(all_patches)

    return all_patches

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

train_a_patch = extract_patches(trainA, sub_patch_dim=(64, 64))
train_b_patch = extract_patches(trainB, sub_patch_dim=(64, 64))

patch_dir_A = './cards_ab/cards_ab/patch_image/A/'
patch_dir_B = './cards_ab/cards_ab/patch_image/B/'
count = 0

for batch in range(0,len(train_a_patch)):
    for num in range(0,6):
        patch_path_a = patch_dir_A + 'A' + str(batch) + '_' + str(count) + '.jpg'
        patch_path_b = patch_dir_B + 'B' + str(batch) + '_' + str(count) + '.jpg'
        count = count+1

        p_img_a = ((train_a_patch[batch][num]+1)/2.0*255).clip(0,255).astype('uint8')
        p_img_b = ((train_b_patch[batch][num] + 1) / 2.0 * 255).clip(0, 255).astype('uint8')

        patch_image_a = Image.fromarray(p_img_a)
        patch_image_b = Image.fromarray(p_img_b)
        patch_image_a.save(patch_path_a, quality=90)
        patch_image_b.save(patch_path_b, quality=90)

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