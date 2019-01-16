from __future__ import print_function

import tensorflow as tf
import numpy as np
import argparse
import time
import glob
import cv2
import os

from matplotlib import pyplot as plt

from tensorflow.layers import (Conv2D, 
                 MaxPooling2D,
                 AveragePooling2D,
                 BatchNormalization,
                 Conv2DTranspose,
                 Flatten,
                 Dense,
                 Dropout)

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
tf.enable_eager_execution(config=config)

class Encoder(tf.keras.Model):
  def __init__(self):
    super(Encoder, self).__init__()
    arg = {
      'activation': tf.nn.relu, 
      'padding': 'same', 
      'kernel_initializer': tf.initializers.random_normal,
      'bias_initializer': tf.initializers.zeros,
      'kernel_regularizer': tf.keras.regularizers.l2(0.1)
    }
    self.conv_11 = Conv2D(name='e_conv_11', filters=64, kernel_size=7, strides=(2,2), **arg)
    self.conv_12 = Conv2D(name='e_conv_12', filters=64, kernel_size=7, strides=(2,2), **arg)
    self.pool_1  = MaxPooling2D(name='e_pool_1', pool_size=4, strides=(2,2), padding='same')
    self.compress_11 = AveragePooling2D(name='e_comp_11', pool_size=5, strides=(3,3), padding='same')
    self.compress_12 = Flatten()
    self.compress_13 = Dense(name='e_comp_13', units=128, activation=tf.nn.relu, use_bias=False,
                             kernel_regularizer=tf.keras.regularizers.l2(0.1))
    self.batch_norm_1 = BatchNormalization(name='e_bn_1')
    self.drop_1  = Dropout(name='e_drop_1', rate=0.5)

    self.conv_21 = Conv2D(name='e_conv_21', filters=128, kernel_size=5, strides=(1,1), **arg)
    self.conv_22 = Conv2D(name='e_conv_22', filters=128, kernel_size=5, strides=(1,1), **arg)
    self.pool_2  = MaxPooling2D(name='e_pool_2', pool_size=4, strides=(2,2), padding='same')
    self.compress_21 = AveragePooling2D(name='e_comp_21', pool_size=5, strides=(3,3), padding='same')
    self.compress_22 = Flatten()
    self.compress_23 = Dense(name='e_comp_23', units=128, activation=tf.nn.relu, use_bias=False,
                             kernel_regularizer=tf.keras.regularizers.l2(0.1))
    self.batch_norm_2 = BatchNormalization(name='e_bn_2')
    self.drop_2  = Dropout(name='e_drop_2', rate=0.5)

    self.conv_31 = Conv2D(name='e_conv_31', filters=256, kernel_size=3, strides=(1,1), **arg)
    self.conv_32 = Conv2D(name='e_conv_32', filters=256, kernel_size=3, strides=(1,1), **arg)
    self.pool_3  = MaxPooling2D(name='e_pool_3', pool_size=2, strides=(2,2), padding='same')
    self.compress_31 = AveragePooling2D(name='e_comp_31', pool_size=3, strides=(1,1), padding='same')
    self.compress_32 = Flatten()
    self.compress_33 = Dense(name='e_comp_33', units=128, activation=tf.nn.relu, use_bias=False,
                             kernel_regularizer=tf.keras.regularizers.l2(0.1))
    self.batch_norm_3 = BatchNormalization(name='e_bn_3')
    self.drop_3  = Dropout(name='e_drop_3', rate=0.5)

    self.batch_norm_4 = BatchNormalization(name='e_bn_4')

  def call(self, x, training=True, verbose=False):
    z = self.conv_11(x);  # print('e_conv_11', z.shape)
    z = self.conv_12(z);  # print('e_conv_12', z.shape)
    z = self.pool_1(z);   # print('e_pool_1', z.shape)
    z = self.batch_norm_1(z, training=training)
    c1 = self.compress_11(z);  # print('e_compress_11', c1.shape)
    c1 = self.compress_12(c1);   # print('e_compress_12', c1.shape)
    c1 = self.compress_13(c1);   # print('e_compress_13', c1.shape)
    z = self.drop_1(z, training=training)

    z = self.conv_21(z);   # print('e_conv_22', z.shape)
    z = self.conv_22(z);   # print('e_conv_22', z.shape)
    z = self.pool_2(z);  # print('e_pool_2', z.shape)
    z = self.batch_norm_2(z, training=training)
    c2 = self.compress_21(z);  # print('e_compress_21', c2.shape)
    c2 = self.compress_22(c2);   # print('e_compress_22', c2.shape)
    c2 = self.compress_23(c2);   # print('e_compress_23', c2.shape)
    z = self.drop_2(z, training=training)

    z = self.conv_31(z);   # print('e_conv_12', z.shape)
    z = self.conv_32(z);   # print('e_conv_12', z.shape)
    z = self.pool_3(z);  # print('e_conv_12', z.shape)
    z = self.batch_norm_3(z, training=training)
    c3 = self.compress_31(z);  # print('e_compress_31', c3.shape)
    c3 = self.compress_32(c3);   # print('e_compress_32', c3.shape)
    c3 = self.compress_33(c3);   # print('e_compress_33', c3.shape)

    target_shape = z.shape
    compressed_z = tf.concat([c1, c2, c3], axis=-1)
    compressed_z = self.batch_norm_4(compressed_z, training=training)
    compressed_z = self.drop_3(compressed_z, training=training)

    return compressed_z, target_shape
    

class Decoder(tf.keras.Model):
  def __init__(self):
    super(Decoder, self).__init__()
    arg = {'activation': tf.nn.relu, 'padding': 'same'}

    self.deconv_11 = Conv2DTranspose(name='d_deconv_11', filters=512, kernel_size=(5,5), strides=(2,2), **arg)
    self.deconv_12 = Conv2DTranspose(name='d_deconv_12', filters=512, kernel_size=(5,5), strides=(2,2), **arg)
    self.deconv_13 = Conv2D(name='d_deconv_13', filters=256, kernel_size=(3,3), strides=(1,1), **arg)
    # self.batch_norm_1 = BatchNormalization(name='d_bn_1')
    # self.drop_1  = Dropout(name='d_drop_1', rate=0.5)

    self.deconv_21 = Conv2DTranspose(name='d_deconv_21', filters=256, kernel_size=(5,5), strides=(2,2), **arg)
    self.deconv_22 = Conv2DTranspose(name='d_deconv_22', filters=256, kernel_size=(5,5), strides=(2,2), **arg)
    self.deconv_23 = Conv2D(name='d_deconv_23', filters=128, kernel_size=(3,3), strides=(1,1), **arg)
    # self.batch_norm_2 = BatchNormalization(name='d_bn_2')
    # self.drop_2  = Dropout(name='d_drop_2', rate=0.5)

    self.deconv_31 = Conv2DTranspose(name='d_deconv_31', filters=128, kernel_size=(5,5), strides=(2,2), **arg)
    self.deconv_32 = Conv2D(name='d_deconv_32', filters=64, kernel_size=(3,3), strides=(1,1), **arg)
    self.deconv_33 = Conv2D(name='d_deconv_33', filters=64, kernel_size=(3,3), strides=(1,1), **arg)
    # self.batch_norm_3 = BatchNormalization(name='d_bn_3')
    # self.drop_3  = Dropout(name='d_drop_3', rate=0.5)

    self.deconv_4  = Conv2D(name='d_deconv_4', filters=3, kernel_size=(3,3), strides=(1,1), padding='same')

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

class LatentDiscriminator(tf.keras.Model):
  def __init__(self):
    super(LatentDiscriminator, self).__init__()
    self.fc_11 = Dense(name='ld_fc_11', units=512, activation=None, use_bias=True)
    self.fc_12 = Dense(name='ld_fc_12', units=512, activation=tf.nn.sigmoid, use_bias=True)
    self.drop_1 = Dropout(name='ld_drop_1', rate=0.5)
    self.fc_21 = Dense(name='ld_fc_21', units=256, activation=None, use_bias=True)
    self.fc_22 = Dense(name='ld_fc_22', units=256, activation=tf.nn.sigmoid, use_bias=True)
    self.drop_2 = Dropout(name='ld_drop_2', rate=0.5)
    self.fc_3 = Dense(name='ld_fc_3', units=128, activation=tf.nn.sigmoid, use_bias=True)
    self.classifier = Dense(name='ld_classifier', units=2, activation=tf.nn.softmax, use_bias=False)

  def call(self, zin, training=True, verbose=False):
    z = self.fc_11(zin)
    z = self.fc_12(z)
    z = self.drop_1(z, training=training)
    z = self.fc_21(z)
    z = self.fc_22(z)
    z = self.drop_2(z, training=training)
    z = self.fc_3(z)
    logit = self.classifier(z)
    return logit

class Discriminator(tf.keras.Model):
  def __init__(self):
    super(Discriminator, self).__init__()
    arg = {'activation': tf.nn.relu, 'padding': 'same'}
    self.conv_11 = Conv2D(name='di_conv_11', filters=64, kernel_size=(5,5), strides=(2,2), **arg)
    self.conv_12 = Conv2D(name='di_conv_12', filters=64, kernel_size=(3,3), strides=(1,1), **arg)
    self.conv_13 = Conv2D(name='di_conv_13', filters=64, kernel_size=(3,3), strides=(1,1), **arg)
    self.pool_1 = MaxPooling2D(name='di_pool_1', pool_size=(5,5), strides=(2,2), padding='same')
    self.drop_1 = Dropout(0.5)

    self.conv_21 = Conv2D(name='di_conv_21', filters=128, kernel_size=(3,3), strides=(1,1), **arg)
    self.conv_22 = Conv2D(name='di_conv_22', filters=128, kernel_size=(3,3), strides=(1,1), **arg)
    self.conv_23 = Conv2D(name='di_conv_23', filters=128, kernel_size=(3,3), strides=(1,1), **arg)
    self.pool_2 = MaxPooling2D(name='di_pool_2', pool_size=(3,3), strides=(2,2), padding='same')
    self.drop_2 = Dropout(0.5)

    self.conv_31 = Conv2D(name='di_conv_31', filters=256, kernel_size=(3,3), strides=(2,2), **arg)
    self.conv_32 = Conv2D(name='di_conv_32', filters=256, kernel_size=(3,3), strides=(1,1), **arg)
    self.conv_33 = Conv2D(name='di_conv_33', filters=256, kernel_size=(3,3), strides=(1,1), **arg)
    self.pool_3 = MaxPooling2D(name='di_pool_3', pool_size=(3,3), strides=(2,2), padding='same')
    self.drop_3 = Dropout(0.5)

    self.flattener = Flatten()
    self.drop_4 = Dropout(0.5)
    self.classifier_1 = Dense(name='di_cls_1', units=512, activation=tf.nn.relu, use_bias=True)
    self.drop_5 = Dropout(0.5)
    self.classifier_2 = Dense(name='di_cls_2', units=256, activation=tf.nn.relu, use_bias=True)
    self.classifier_3 = Dense(name='di_cls_3', units=2, activation=None, use_bias=True)
    
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
    z = self.drop_5(z, training=training)
    z = self.classifier_2(z)
    logit = self.classifier_3(z)

    return logit

class Autoencoder(tf.keras.Model):
  def __init__(self):
    super(Autoencoder, self).__init__()
    self.encoder = Encoder()
    # self.mu_layer = Dense(name='mu', units=1, use_bias=False)
    # self.log_var_layer = Dense(name='log_var', units=1, use_bias=False)
    self.decompress_1 = Dense(name='d_decompress_1', units=1024, activation=tf.nn.relu)
    self.decompress_2 = Dense(name='d_decompress_2', units=1024, activation=tf.nn.relu)

    # self.classifier_1 = Dense(name='ae_class_1', units = 256, activation=tf.nn.relu)
    # self.drop_1 = Dropout(0.5)
    # self.classifier_2 = Dense(name='ae_class_2', units = 256, activation=tf.nn.relu)
    # self.classifier_3 = Dense(name='ae_class_3', units = 2, activation=None)
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
    #   return xhat, mu, log_var, z
    # else:
    #   return xhat, mu, log_var

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

  # fig, axs = plt.subplots(5,5, figsize=(5,5))
  for x_, ax in zip(xhat, axs.flatten()):
    ax.imshow(x_)
    ax.set_xticks([])
    ax.set_yticks([])

  fig.savefig('{}_xhat.png'.format(savebase), bbox_inches='tight')

def train_vae(x, model, discr, ldiscr, optimizer, saver, yfake_onehot, return_loss='all'):
  n_samples = x.shape[0]
  batch_idx = np.random.choice(n_samples, args.batch)
  batch_x = x[batch_idx, ...]
  batch_x = (batch_x / 255.).astype(np.float32)
  # batch_y = np.eye(BATCH, 2)[y[batch_idx]]

  # Generator training
  with tf.GradientTape() as tape:
    # xhat, mu, log_var = model(tf.constant(batch_x, dtype=tf.float32))
    xhat, zhat = model(tf.constant(batch_x, dtype=tf.float32), return_z=True)

    mse = tf.losses.mean_squared_error(labels=batch_x, predictions=xhat, 
      reduction=tf.losses.Reduction.NONE)
    mse = tf.reduce_sum(mse, axis=[1,2,3])
    # kld = -0.5 * tf.reduce_sum(1 + log_var - tf.square(mu) - tf.exp(log_var), 1)

    yfake_image = discr(xhat)
    dloss = tf.losses.softmax_cross_entropy(onehot_labels=yfake_onehot, 
                                            logits=yfake_image)

    yfake_latent = ldiscr(zhat)    
    ldloss = tf.losses.softmax_cross_entropy(onehot_labels=yfake_onehot, 
                                             logits=yfake_latent)

    reg_loss = tf.reduce_mean(model.losses)
    loss = mse + (2*dloss) + (2*ldloss) + reg_loss

  grads = tape.gradient(loss, model.variables)
  optimizer.apply_gradients(zip(grads, model.variables))

  if return_loss != 'all':
    return loss
  else:
    return batch_x, xhat, zhat, mse, dloss, ldloss, loss

def train_discriminator(batch_x, xhat, discr, optimizer, y_onehot):
  with tf.GradientTape() as tape:
    x_xhat = np.concatenate([batch_x, xhat.numpy()], axis=0)
    yhat = discr(tf.constant(x_xhat, dtype=tf.float32))
    loss = tf.losses.softmax_cross_entropy(onehot_labels=y_onehot, logits=yhat)

  grads = tape.gradient(loss, discr.variables)
  optimizer.apply_gradients(zip(grads, discr.variables))
  return loss

def train_latent_discriminator(zhat, ldiscr, optimizer, y_onehot):
  z_dim = zhat.shape[1]
  with tf.GradientTape() as tape:
    zsample = np.random.normal(size=(args.batch, z_dim))
    zhat = zhat.numpy()
    zsample_zhat = np.concatenate([zsample, zhat], axis=0)
    yhat = ldiscr(tf.constant(zsample_zhat, dtype=tf.float32))
    loss = tf.losses.softmax_cross_entropy(onehot_labels=y_onehot, logits=yhat)

  grads = tape.gradient(loss, ldiscr.variables)
  optimizer.apply_gradients(zip(grads, ldiscr.variables))
  return loss

def load_data():
  # x = np.load('/mnt/slowdata/project_data/va_pnbx/nuclei_images.npy')

  # x1 = np.load('/mnt/slowdata/project_data/va_pnbx/adeno_nuclei.npy')
  # x2 = np.load('/mnt/slowdata/project_data/va_pnbx/nepc_nuclei.npy')
  x1 = np.load('/mnt/linux-data/storage/Dropbox/projects/pnbx/adeno_nuclei.npy')
  x2 = np.load('/mnt/linux-data/storage/Dropbox/projects/pnbx/nepc_nuclei.npy')
  x = np.concatenate([x1, x2], axis=0)

  ## For multi-task learning
  # nepc_ = np.load('./nepc_nuclei_images.npy')
  # x = np.concatenate([adeno_, nepc_], axis=0)
  # nx = x.shape[0]
  # y = np.array([0]*adeno_.shape[0] + [1]*nepc_.shape[0], dtype=np.int)
  # x = (x / 255.).astype(np.float32)
  # return x, y

  print('Numpy dataset loaded: {}'.format(x.shape))
  return x

def main(args):
  xdummy = tf.zeros(shape=(5, 64, 64, 3), dtype=tf.float32)
  model  = Autoencoder()
  discr  = Discriminator()
  ldiscr = LatentDiscriminator()

  _ = model(xdummy, verbose=True)
  model.summary()
  _ = discr(xdummy, verbose=True)
  discr.summary()

  _, zdummy = model(xdummy, return_z=True)
  _ = ldiscr(zdummy, verbose=True)
  ldiscr.summary()

  x = load_data()
  print(x.shape)

  model_optimizer  = tf.train.AdamOptimizer(learning_rate=args.lr)
  discr_optimizer  = tf.train.AdamOptimizer(learning_rate=args.lr)
  ldiscr_optimizer = tf.train.AdamOptimizer(learning_rate=args.lr)
  saver = tf.contrib.eager.Saver(model.variables)

  if args.snapshot is not None:
    saver.restore(args.snapshot)

  # Targets to train the discriminators
  # d_phi(x_real) --> 1
  # d_phi(x_fake) --> 0
  yreal  = np.ones(args.batch, dtype=np.int)
  yfakes   = np.zeros(args.batch, dtype=np.int)
  y_onehot = np.eye(args.batch, 2)[np.concatenate([yreal, yfakes])]

  # Target for the generator to produce real looking images (d_phi(x) --> 1)
  yfake_onehot = tf.constant(np.eye(args.batch, 2)[np.ones(args.batch, dtype=np.int)])

  # Generator head start
  print('Head starting the generator')
  for k in range(args.gen_warmup):
    loss = train_vae(x, model, discr, ldiscr, model_optimizer, saver, yfake_onehot, return_loss=1)
    if k % 250 == 0:
      print('IMG GEN L2 t={:05d}: '.format(k), tf.reduce_mean(loss).numpy())

  # Discriminator catch up
  print('Catching up the image discriminator')
  for k in range(args.img_d_warmup):
    n_samples = x.shape[0]
    batch_idx = np.random.choice(n_samples, args.batch)
    batch_x = x[batch_idx, ...]
    batch_x = (batch_x / 255.).astype(np.float32)
    # xhat, _,_ = model(tf.constant(batch_x, dtype=tf.float32))
    xhat = model(tf.constant(batch_x, dtype=tf.float32))
    loss = train_discriminator(batch_x, xhat, discr, discr_optimizer, y_onehot)
    if k % 250 == 0:
      print('IMG DISCR LOSS t={:05d}: {:3.5f}'.format(k, loss.numpy()))

  print('Catching up the image discriminator')
  for k in range(args.latent_d_warmup):
    n_samples = x.shape[0]
    batch_idx = np.random.choice(n_samples, args.batch)
    batch_x = x[batch_idx, ...]
    batch_x = (batch_x / 255.).astype(np.float32)
    xhat, zhat = model(tf.constant(batch_x, dtype=tf.float32), return_z=True)
    loss = train_latent_discriminator(zhat, ldiscr, ldiscr_optimizer, y_onehot)
    if k % 250 == 0:
      print('LAT DISCR LOSS t={:05d}: {:3.5f}'.format(k, loss.numpy()))

  print('Main training procedure')
  fig, axs = plt.subplots(5,5, figsize=(5,5))
  for k in range(args.steps):
    batch_x, xhat, zhat, mse, dloss, ldloss, loss = \
      train_vae(x, model, discr, ldiscr, model_optimizer, saver, yfake_onehot)

    if k % args.print_steps == 0: 
      print('Step: {}\t'
          'Ls:{: 4.3f}\t'
          'MSE:{: 4.3f}\t'
          'DL:{: 3.3f}\t'
          'LL:{: 3.3f}\t'
          'Mn:{: 3.3f}\t'
          'Sd:{: 3.3f}'.format(k,
                     tf.reduce_mean(loss).numpy(),
                     tf.reduce_mean(mse).numpy(),
                     tf.reduce_mean(dloss).numpy(),
                     tf.reduce_mean(ldloss).numpy(),
                     np.mean(zhat.numpy()),
                     np.std(zhat.numpy()))
      )

      # Recycle the generated images and latent variables to train the discriminators
      for _ in range(args.d_train):
        img_d_loss = train_discriminator(batch_x, xhat, discr, discr_optimizer, y_onehot)
        lat_d_loss = train_latent_discriminator(zhat, ldiscr, ldiscr_optimizer, y_onehot)

      print('\timg discr: {:3.5f}'.format(tf.reduce_mean(img_d_loss).numpy()))
      print('\tlat discr: {:3.5f}'.format(tf.reduce_mean(lat_d_loss).numpy()))

    if k % args.draw_steps == 0:
      savebase = 'autoencoder_debug/{:08d}'.format(k)
      draw_result(batch_x, xhat.numpy(), fig, axs, savebase=savebase)

    if k % args.save_steps == 0:
      saver.save('./autoencoder_model/autoencoder', global_step=k)

if __name__ == '__main__':
  parser = argparse.ArgumentParser()

  parser.add_argument('--lr', default=1e-4, type=float)
  parser.add_argument('--batch', default=32, type=int)
  parser.add_argument('--steps', default=250000, type=int)
  parser.add_argument('--d_train', default=1, type=int)
  parser.add_argument('--snapshot', default=None, type=str)
  parser.add_argument('--gen_warmup', default=2000, type=int)
  parser.add_argument('--draw_steps', default=1000, type=int)
  parser.add_argument('--save_steps', default=2500, type=int)
  parser.add_argument('--print_steps', default=25, type=int)
  parser.add_argument('--img_d_warmup', default=500, type=int)
  parser.add_argument('--latent_d_warmup', default=500, type=int)

  args = parser.parse_args()
  main(args)
