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

def draw_result(x, xhat, savebase=None, n=25):
    print('x:', x.shape, 'xhat:', xhat.shape)
    fig, axs = plt.subplots(5,5, figsize=(5,5))
    n_x = x.shape[0]-1
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

def main(args):
    if args.src_npy is None:
        print('Supply src_npy')
        return 0
    if args.out_npy is None:
        print('Supply out_npy')
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
    
    all_feat = []
    for k, batch in enumerate(image_batches):
        batch = (batch / 255.).astype(np.float32)
        batch_hat, features = model(tf.constant(batch, dtype=tf.float32), return_z=True)
        all_feat.append(features)

        if k % 100 == 0:
            print('batch {:06d}'.format(k))

        # if k % 50 == 0 a:
        #     draw_result(x=batch, xhat=batch_hat.numpy(), 
        #                 savebase='nepc_{:05d}'.format(k))

    all_feat = np.concatenate(all_feat, axis=0)
    print('all_feat', all_feat.shape)

    np.save(args.out_npy, all_feat)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--src_npy', default=None, type=str)
    parser.add_argument('--out_npy', default=None, type=str)
    parser.add_argument('--snapshot', 
        type = str,
        default = './autoencoder_model/autoencoder-5000')
    parser.add_argument('--batch', default=48, type=int)

    args = parser.parse_args()

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    tf.enable_eager_execution(config=config)
    main(args)