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
from crappyhist import crappyhist

def main(args):
  if args.src_npy is None:
    print('Supply src_npy')
    return 0

  model = Autoencoder()
  dummyx = tf.zeros((5, 64, 64, 3), dtype=tf.float32)
  _ = model(dummyx, verbose=True)
  saver = tfe.Saver(model.variables)
  saver.restore(args.snapshot)
  model.summary()

  nuclei =np.load(args.src_npy)
  print(nuclei.shape, nuclei.dtype, nuclei.min(), nuclei.max())
  n_images = nuclei.shape[0]
  n_batches = n_images // args.batch
  image_batches = np.array_split(nuclei, n_batches)[:50]
  
  for k, batch in enumerate(image_batches):
    batch = (batch / 255.).astype(np.float32)
    batch_hat, features = model(tf.constant(batch, dtype=tf.float32), return_z=True, training=False)

    crappyhist(features)
    inp = input('Continue')
    # if k % 50 == 0 a:
    #     draw_result(x=batch, xhat=batch_hat.numpy(), 
    #                 savebase='nepc_{:05d}'.format(k))


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--src_npy', default=None, type=str)
  parser.add_argument('--snapshot', 
      default = './autoencoder_model/autoencoder-77500',
      type = str,)
  parser.add_argument('--batch', default=48, type=int)

  args = parser.parse_args()

  config = tf.ConfigProto()
  config.gpu_options.allow_growth = True

  tf.enable_eager_execution(config=config)
  main(args)