from __future__ import print_function

import tensorflow as tf
import numpy as np
import argparse
import glob
import cv2
import os

from matplotlib import pyplot as plt

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
tf.enable_eager_execution(config=config)

class Encoder(tf.keras.Model):
    def __init__(self):
        super(Encoder, self).__init__()
        arg = {'activation': tf.nn.relu, 'padding': 'same'}
        self.conv_11 = tf.layers.Conv2D(name='e_conv_11', filters=64, kernel_size=7, strides=(2,2), **arg)
        self.conv_12 = tf.layers.Conv2D(name='e_conv_12', filters=64, kernel_size=7, strides=(2,2), **arg)
        self.pool_1  = tf.layers.MaxPooling2D(name='e_pool_1', pool_size=4, strides=(2,2), padding='same')
        self.compress_11 = tf.layers.AveragePooling2D(name='e_comp_11', pool_size=5, strides=(3,3), padding='same')
        self.compress_12 = tf.layers.Flatten()
        self.compress_13 = tf.layers.Dense(name='e_comp_13', units=128, activation=tf.nn.relu)
        self.batch_norm_1 = tf.layers.BatchNormalization(name='e_bn_1')
        self.drop_1  = tf.layers.Dropout(name='e_drop_1', rate=0.5)

        self.conv_21 = tf.layers.Conv2D(name='e_conv_21', filters=128, kernel_size=5, strides=(1,1), **arg)
        self.conv_22 = tf.layers.Conv2D(name='e_conv_22', filters=128, kernel_size=5, strides=(1,1), **arg)
        self.pool_2  = tf.layers.MaxPooling2D(name='e_pool_2', pool_size=4, strides=(2,2), padding='same')
        self.compress_21 = tf.layers.AveragePooling2D(name='e_comp_21', pool_size=5, strides=(3,3), padding='same')
        self.compress_22 = tf.layers.Flatten()
        self.compress_23 = tf.layers.Dense(name='e_comp_23', units=128, activation=tf.nn.relu)
        self.batch_norm_2 = tf.layers.BatchNormalization(name='e_bn_2')
        self.drop_2  = tf.layers.Dropout(name='e_drop_2', rate=0.5)

        self.conv_31 = tf.layers.Conv2D(name='e_conv_31', filters=256, kernel_size=3, strides=(1,1), **arg)
        self.conv_32 = tf.layers.Conv2D(name='e_conv_32', filters=256, kernel_size=3, strides=(1,1), **arg)
        self.pool_3  = tf.layers.MaxPooling2D(name='e_pool_3', pool_size=2, strides=(2,2), padding='same')
        self.compress_31 = tf.layers.AveragePooling2D(name='e_comp_31', pool_size=3, strides=(1,1), padding='same')
        self.compress_32 = tf.layers.Flatten()
        self.compress_33 = tf.layers.Dense(name='e_comp_33', units=128, activation=tf.nn.relu)
        self.batch_norm_3 = tf.layers.BatchNormalization(name='e_bn_3')
        self.drop_3  = tf.layers.Dropout(name='e_drop_3', rate=0.5)

    def call(self, x, training=True, verbose=False):
        z = self.conv_11(x);    # print('e_conv_11', z.shape)
        z = self.conv_12(z);    # print('e_conv_12', z.shape)
        z = self.pool_1(z);     # print('e_pool_1', z.shape)
        z = self.batch_norm_1(z, training=training)
        c1 = self.compress_11(z);    # print('e_compress_11', c1.shape)
        c1 = self.compress_12(c1);   # print('e_compress_12', c1.shape)
        c1 = self.compress_13(c1);   # print('e_compress_13', c1.shape)
        z = self.drop_1(z, training=training)

        z = self.conv_21(z);   # print('e_conv_22', z.shape)
        z = self.conv_22(z);   # print('e_conv_22', z.shape)
        z = self.pool_2(z);    # print('e_pool_2', z.shape)
        z = self.batch_norm_2(z, training=training)
        c2 = self.compress_21(z);    # print('e_compress_21', c2.shape)
        c2 = self.compress_22(c2);   # print('e_compress_22', c2.shape)
        c2 = self.compress_23(c2);   # print('e_compress_23', c2.shape)
        z = self.drop_2(z, training=training)

        z = self.conv_31(z);   # print('e_conv_12', z.shape)
        z = self.conv_32(z);   # print('e_conv_12', z.shape)
        z = self.pool_3(z);    # print('e_conv_12', z.shape)
        z = self.batch_norm_3(z, training=training)
        c3 = self.compress_31(z);    # print('e_compress_31', c3.shape)
        c3 = self.compress_32(c3);   # print('e_compress_32', c3.shape)
        c3 = self.compress_33(c3);   # print('e_compress_33', c3.shape)

        target_shape = z.shape
        compressed_z = tf.concat([c1, c2, c3], axis=-1)
        compressed_z = self.drop_3(compressed_z, training=training)

        return compressed_z, target_shape
        

class Decoder(tf.keras.Model):
    def __init__(self):
        super(Decoder, self).__init__()
        arg = {'activation': tf.nn.relu, 'padding': 'same'}

        self.deconv_11 = tf.layers.Conv2DTranspose(name='d_deconv_11', filters=512, kernel_size=(5,5), strides=(2,2), **arg)
        self.deconv_12 = tf.layers.Conv2DTranspose(name='d_deconv_12', filters=512, kernel_size=(5,5), strides=(2,2), **arg)
        self.deconv_13 = tf.layers.Conv2D(name='d_deconv_13', filters=256, kernel_size=(3,3), strides=(1,1), **arg)
        # self.batch_norm_1 = tf.layers.BatchNormalization(name='d_bn_1')
        # self.drop_1    = tf.layers.Dropout(name='d_drop_1', rate=0.5)

        self.deconv_21 = tf.layers.Conv2DTranspose(name='d_deconv_21', filters=256, kernel_size=(5,5), strides=(2,2), **arg)
        self.deconv_22 = tf.layers.Conv2DTranspose(name='d_deconv_22', filters=256, kernel_size=(5,5), strides=(2,2), **arg)
        self.deconv_23 = tf.layers.Conv2D(name='d_deconv_23', filters=128, kernel_size=(3,3), strides=(1,1), **arg)
        # self.batch_norm_2 = tf.layers.BatchNormalization(name='d_bn_2')
        # self.drop_2    = tf.layers.Dropout(name='d_drop_2', rate=0.5)

        self.deconv_31 = tf.layers.Conv2DTranspose(name='d_deconv_31', filters=128, kernel_size=(5,5), strides=(2,2), **arg)
        self.deconv_32 = tf.layers.Conv2D(name='d_deconv_32', filters=64, kernel_size=(3,3), strides=(1,1), **arg)
        self.deconv_33 = tf.layers.Conv2D(name='d_deconv_33', filters=64, kernel_size=(3,3), strides=(1,1), **arg)
        # self.batch_norm_3 = tf.layers.BatchNormalization(name='d_bn_3')
        # self.drop_3    = tf.layers.Dropout(name='d_drop_3', rate=0.5)

        self.deconv_4  = tf.layers.Conv2D(name='d_deconv_4', filters=3, kernel_size=(3,3), strides=(1,1), padding='same')

    def call(self, z, training=True, verbose=False):
        z = self.deconv_11(z);   # print('d_deconv_11', z.shape)
        z = self.deconv_12(z);   # print('d_deconv_12', z.shape)
        z = self.deconv_13(z);   # print('d_deconv_12', z.shape)
        # z = self.batch_norm_1(z, training=training)
        # z = self.drop_1(z, training=training)

        z = self.deconv_21(z);   # print('d_deconv_21', z.shape)
        z = self.deconv_22(z);   # print('d_deconv_22', z.shape)
        z = self.deconv_23(z);   # print('d_deconv_22', z.shape)
        # z = self.batch_norm_2(z, training=training)
        # z = self.drop_2(z, training=training)

        z = self.deconv_31(z);   # print('d_deconv_31', z.shape)
        z = self.deconv_32(z);   # print('d_deconv_32', z.shape)
        z = self.deconv_33(z);   # print('d_deconv_32', z.shape)
        # z = self.batch_norm_3(z, training=training)
        # x = self.drop_3(z, training=training)

        x = self.deconv_4(z);   # print('d_deconv_4', x.shape)
        return x


class Discriminator(tf.keras.Model):
    def __init__(self):
        super(Discriminator, self).__init__()
        arg = {'activation': tf.nn.relu, 'padding': 'same'}
        self.conv_11 = tf.layers.Conv2D(name='di_conv_11', filters=64, kernel_size=(5,5), strides=(2,2), **arg)
        self.conv_12 = tf.layers.Conv2D(name='di_conv_12', filters=64, kernel_size=(3,3), strides=(1,1), **arg)
        self.conv_13 = tf.layers.Conv2D(name='di_conv_13', filters=64, kernel_size=(3,3), strides=(1,1), **arg)
        self.pool_1 = tf.layers.MaxPooling2D(name='di_pool_1', pool_size=(5,5), strides=(2,2), padding='same')
        self.drop_1 = tf.layers.Dropout(0.5)

        self.conv_21 = tf.layers.Conv2D(name='di_conv_21', filters=128, kernel_size=(3,3), strides=(1,1), **arg)
        self.conv_22 = tf.layers.Conv2D(name='di_conv_22', filters=128, kernel_size=(3,3), strides=(1,1), **arg)
        self.conv_23 = tf.layers.Conv2D(name='di_conv_23', filters=128, kernel_size=(3,3), strides=(1,1), **arg)
        self.pool_2 = tf.layers.MaxPooling2D(name='di_pool_2', pool_size=(3,3), strides=(2,2), padding='same')
        self.drop_2 = tf.layers.Dropout(0.5)

        self.conv_31 = tf.layers.Conv2D(name='di_conv_31', filters=256, kernel_size=(3,3), strides=(2,2), **arg)
        self.conv_32 = tf.layers.Conv2D(name='di_conv_32', filters=256, kernel_size=(3,3), strides=(1,1), **arg)
        self.conv_33 = tf.layers.Conv2D(name='di_conv_33', filters=256, kernel_size=(3,3), strides=(1,1), **arg)
        self.pool_3 = tf.layers.MaxPooling2D(name='di_pool_3', pool_size=(3,3), strides=(2,2), padding='same')
        self.drop_3 = tf.layers.Dropout(0.5)

        self.flattener = tf.layers.Flatten()
        self.drop_4 = tf.layers.Dropout(0.5)
        self.classifier_1 = tf.layers.Dense(name='di_cls_1', units=256, activation=tf.nn.relu, use_bias=True)
        self.classifier_2 = tf.layers.Dense(name='di_cls_2', units=2, activation=None, use_bias=False)
        
    def call(self, x, training=True, verbose=False):
        z = self.conv_11(x)
        z = self.conv_12(z)
        z = self.conv_13(z)
        z = self.pool_1(z)
        z = self.drop_1(z, training=training)
        if verbose:
            print('z', z.shape)
        
        z = self.conv_21(z)
        z = self.conv_22(z)
        z = self.conv_23(z)
        z = self.pool_2(z)
        z = self.drop_2(z, training=training)
        if verbose:
            print('z', z.shape)

        z = self.conv_31(z)
        z = self.conv_32(z)
        z = self.conv_33(z)
        z = self.pool_3(z)
        z = self.drop_3(z, training=training)
        if verbose:
            print('z', z.shape)

        z = self.flattener(z)
        if verbose:
            print('z flat', z.shape)

        z = self.drop_4(z, training=training)
        z = self.classifier_1(z)
        logit = self.classifier_2(z)

        return logit


class Autoencoder(tf.keras.Model):
    def __init__(self):
        super(Autoencoder, self).__init__()
        self.encoder = Encoder()
        # self.mu_layer = tf.layers.Dense(name='mu', units=1, use_bias=False)
        # self.log_var_layer = tf.layers.Dense(name='log_var', units=1, use_bias=False)
        self.decompress_1 = tf.layers.Dense(name='d_decompress_1', units=1024, activation=tf.nn.relu)
        self.decompress_2 = tf.layers.Dense(name='d_decompress_2', units=1024, activation=tf.nn.relu)

        # self.classifier_1 = tf.layers.Dense(name='ae_class_1', units = 256, activation=tf.nn.relu)
        # self.drop_1 = tf.layers.Dropout(0.5)
        # self.classifier_2 = tf.layers.Dense(name='ae_class_2', units = 256, activation=tf.nn.relu)
        # self.classifier_3 = tf.layers.Dense(name='ae_class_3', units = 2, activation=None)
        self.decoder = Decoder()

    def call(self, x, z_in=None, training=True, return_z=False, verbose=False):
        # print('x in', x.shape)
        z, target_shape = self.encoder(x, training=training, verbose=verbose)
        n_target_units = np.prod(target_shape[1:])
        # print('encoder z', z.shape)
        # print('n target units', n_target_units)

        # Reparamaterization trick
        # mu = self.mu_layer(z);   # print('mu', mu.shape)
        # log_var = self.log_var_layer(z);  # print('log_var', log_var.shape)
        # batch = x.shape[0].value
        # dim = z.shape[-1].value
        # eps = tf.random_normal(shape=[batch, dim], mean=0, stddev=1.0)
        # z_sample = mu + eps * tf.exp(0.5 * log_var)

        z_sample = self.decompress_1(z);  # print('z_sample', z_sample.shape)
        z_sample = self.decompress_2(z);  # print('z_sample', z_sample.shape)
        z_sample = tf.reshape(z_sample, shape=target_shape);  # print('z_sample', z_sample.shape)

        xhat = self.decoder(z_sample, training=training);  # print('xhat', xhat.shape)

        # Predict yhat 
        # yhat = self.classifier_1(z)
        # yhat = self.drop_1(yhat, training=training)
        # yhat = self.classifier_2(yhat)
        # yhat = self.classifier_3(yhat)

        # if return_z:
        #     return xhat, mu, log_var, z
        # else:
        #     return xhat, mu, log_var

        if return_z:
            return xhat, z
        else:
            return xhat             



def draw_result(x, xhat, fig, axs, savebase=None, n=25):
    print('x:', x.shape, 'xhat:', xhat.shape)
    # fig, axs = plt.subplots(5,5, figsize=(5,5))
    n_x = x.shape[0]
    choose_x = np.random.choice(range(n_x), n, replace=False)
    print('choose_x', choose_x)
    x = x[choose_x, ...]
    xhat = xhat[choose_x, ...]
    for x_, ax in zip(x, axs.flatten()):
        ax.imshow(x_)
        ax.set_xticks([])
        ax.set_yticks([])

    fig.savefig('{}_x.png'.format(savebase), bbox_inches='tight')

    fig, axs = plt.subplots(5,5, figsize=(5,5))
    for x_, ax in zip(xhat, axs.flatten()):
        ax.imshow(x_)
        ax.set_xticks([])
        ax.set_yticks([])

    fig.savefig('{}_xhat.png'.format(savebase), bbox_inches='tight')


BATCH = 96
yfake_onehot = tf.constant(np.eye(BATCH, 2)[np.ones(BATCH, dtype=np.int)])
def train_vae(x, model, discr, optimizer, saver):
    n_samples = x.shape[0]
    batch_idx = np.random.choice(n_samples, BATCH)
    batch_x = x[batch_idx, ...]
    batch_x = (batch_x / 255.).astype(np.float32)
    # batch_y = np.eye(BATCH, 2)[y[batch_idx]]

    # Generator training
    with tf.GradientTape() as tape:
        # xhat, mu, log_var = model(tf.constant(batch_x, dtype=tf.float32))
        xhat = model(tf.constant(batch_x, dtype=tf.float32))

        l2 = tf.losses.mean_squared_error(labels=batch_x, predictions=xhat, 
            reduction=tf.losses.Reduction.NONE)
        l2 = tf.reduce_sum(l2, axis=[1,2,3])
        # kld = -0.5 * tf.reduce_sum(1 + log_var - tf.square(mu) - tf.exp(log_var), 1)

        yfake_hat = discr(xhat)
        dl = tf.losses.softmax_cross_entropy(onehot_labels=yfake_onehot, 
            logits=yfake_hat)

        # cl = tf.losses.softmax_cross_entropy(onehot_labels=batch_y, 
        #     logits=yhat)

        # loss = l2 + kld + (5*dl)
        # loss = l2 + (5*dl) + (10*cl)
        loss = l2 + (5*dl)

    grads = tape.gradient(loss, model.variables)
    optimizer.apply_gradients(zip(grads, model.variables))
    # return batch_x, xhat, l2, kld, dl
    # return batch_x, xhat, l2, dl, cl
    return batch_x, xhat, l2, dl


yreal = np.ones(BATCH, dtype=np.int)
yfakes = np.zeros(BATCH, dtype=np.int)
y_onehot = np.eye(BATCH, 2)[np.concatenate([yreal, yfakes])]
def train_discriminator(batch_x, xhat, discr, optimizer):
    with tf.GradientTape() as tape:
        x_xhat = np.concatenate([batch_x, xhat.numpy()], axis=0)
        yhat = discr(tf.constant(x_xhat, dtype=tf.float32))

        loss = tf.losses.softmax_cross_entropy(onehot_labels=y_onehot, logits=yhat)

    grads = tape.gradient(loss, discr.variables)
    optimizer.apply_gradients(zip(grads, discr.variables))

    return loss


def load_data():
    x = np.load('/mnt/slowdata/project_data/va_pnbx/adeno_nuclei_images.npy')
    # nepc_ = np.load('./nepc_nuclei_images.npy')
    # x = np.concatenate([adeno_, nepc_], axis=0)
    # nx = x.shape[0]
    # y = np.array([0]*adeno_.shape[0] + [1]*nepc_.shape[0], dtype=np.int)
    # x = (x / 255.).astype(np.float32)
    # return x, y
    return x


def main(args):
    xdummy = tf.zeros(shape=(5, 64, 64, 3), dtype=tf.float32)
    model = Autoencoder()
    discr = Discriminator()

    _ = model(xdummy, verbose=True)
    model.summary()
    _ = discr(xdummy, verbose=True)
    discr.summary()

    # x = np.load(args.src)
    # x = (x / 255.).astype(np.float32)
    # x, y = load_data()
    x = load_data()
    print(x.shape)

    model_optimizer = tf.train.AdamOptimizer(learning_rate=1e-4)
    discr_optimizer = tf.train.AdamOptimizer(learning_rate=1e-4)
    saver = tf.contrib.eager.Saver(model.variables)

    if args.snapshot is not None:
        saver.restore(args.snapshot)

    # Generator head start
    print('Head starting the generator')
    for k in range(2500):
        train_vae(x, model, discr, model_optimizer, saver)
        if k % 250 == 0:
            print('step {:06d}'.format(k))

    # Discriminator catch up
    print('Catching up the discriminator')
    for k in range(500):
        n_samples = x.shape[0]
        batch_idx = np.random.choice(n_samples, BATCH)
        batch_x = x[batch_idx, ...]
        batch_x = (batch_x / 255.).astype(np.float32)
        # xhat, _,_ = model(tf.constant(batch_x, dtype=tf.float32))
        xhat = model(tf.constant(batch_x, dtype=tf.float32))
        train_discriminator(batch_x, xhat, discr, discr_optimizer)
        if k % 250 == 0:
            print('step {:06d}'.format(k))

    print('Main training procedure')
    fig, axs = plt.subplots(5,5, figsize=(5,5))
    for k in range(200000):
        # batch_x, xhat, l2, kld, dl = train_vae(x, model, discr, model_optimizer, saver)
        batch_x, xhat, l2, dl = train_vae(x, model, discr, model_optimizer, saver)
        if k % 10 == 0: 
            print(k, 'l2: ', tf.reduce_mean(l2).numpy(), 
                    #  'kld: ', tf.reduce_mean(kld).numpy(),
                     'dl: ', tf.reduce_mean(dl).numpy(),
                    #  'cl: ', tf.reduce_mean(cl).numpy()
                     )

        if k % 10 == 0: 
            loss = train_discriminator(batch_x, xhat, discr, discr_optimizer)
            print(k, 'discriminator: ', tf.reduce_mean(loss).numpy())

        if k % 250 == 0:
            savebase = 'autoencoder_debug/{:08d}'.format(k)
            draw_result(batch_x, xhat.numpy(), fig, axs, savebase=savebase)

        if k % 2500 == 0:
            saver.save('./autoencoder_model/autoencoder', global_step=k)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # parser.add_argument('--src', default='./nuclei_dump.npy', type=str)
    parser.add_argument('--snapshot', default=None, type=str)

    args = parser.parse_args()
    main(args)