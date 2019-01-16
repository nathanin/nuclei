from __future__ import print_function

import tensorflow as tf
import tensorflow.contrib.eager as tfe
import numpy as np
import argparse
import glob
import cv2
import os

from matplotlib import pyplot as plt 
from autoencoder import Autoencoder

def draw_result(x, xhat, fig, axs, savebase=None, n=25):
  print('choosing from x:', x.shape, x.min(), x.max(), 
                   'xhat:', xhat.shape, xhat.min(), xhat.max())
  
  n_x = x.shape[0]
  choose_x = np.asarray(np.random.choice(range(n_x), n, replace=False))
  x = x[choose_x, ...]
  xhat = xhat[choose_x, ...]
  print('drawing from x:', x.shape, 'xhat:', xhat.shape)
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

def main(args):
  if args.src_npy is None:
    print('Supply src_npy')
    return 0
  if args.dst_npy is None:
    print('Supply dst_npy')
    return 0

  model = Autoencoder()
  dummyx = tf.zeros((5, 64, 64, 3), dtype=tf.float32)
  _ = model(dummyx, verbose=True)
  saver = tfe.Saver(model.variables)
  saver.restore(args.snapshot)
  model.summary()

  nuclei = np.load(args.src_npy)
  print(nuclei.shape, nuclei.dtype, nuclei.min(), nuclei.max())

  if args.shuffle:
    print('Shuffling')
    np.random.shuffle(nuclei)

  n_images = nuclei.shape[0]
  n_batches = n_images // args.batch

  nuclei = np.array_split(nuclei, n_batches)
  print('Split into {} batches'.format(len(nuclei)))

  if args.n_batches is not None:
    subset_batches = min(n_batches, args.n_batches)
    print('Subsetting {} batches'.format(args.n_batches))
    nuclei = nuclei[:subset_batches]
  
  if args.draw:
    fig , axs = plt.subplots(5,5, figsize=(5,5))

  all_feat = []
  for k, batch in enumerate(nuclei):
    batch = (batch / 255.).astype(np.float32)
    batch_hat, features = model(tf.constant(batch, dtype=tf.float32), return_z=True, training=False)
    all_feat.append(features)

    if k % 50 == 0:
      print('batch {:06d}'.format(k))

    if args.draw:
      if k % 10 == 0:
        savebase = os.path.join(args.save, '{:05d}'.format(k))
        draw_result(batch, batch_hat.numpy(), fig, axs, savebase=savebase)

  all_feat = np.concatenate(all_feat, axis=0)
  print('all_feat', all_feat.shape)

  np.save(args.dst_npy, all_feat)

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('src_npy', default=None, type=str)
  parser.add_argument('dst_npy', default=None, type=str)
  parser.add_argument('--snapshot', 
    type = str,
    default = './autoencoder_model/autoencoder-5000')

  # Optional stuff
  parser.add_argument('--seed', default=None, type=int)
  parser.add_argument('--draw', default=False, action='store_true')
  parser.add_argument('--save', default='autoencoder_test/', type=str)
  parser.add_argument('--batch', default=48, type=int)
  parser.add_argument('--shuffle', default=False, action='store_true')
  parser.add_argument('--n_batches', default=None, type=int)

  args = parser.parse_args()

  config = tf.ConfigProto()
  config.gpu_options.allow_growth = True

  tf.enable_eager_execution(config=config)
  main(args)