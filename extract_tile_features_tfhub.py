from __future__ import print_function

import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import argparse
import glob
import cv2
import os

def load_batch(batch, height, width):
    imgs = []
    for b in batch:
        img = cv2.imread(b)[:,:,::-1] / 255.
        img = cv2.resize(img, dsize=(height, width))
        imgs.append(np.expand_dims(img, 0))

    imgs = np.concatenate(imgs, axis=0)
    return imgs


def save_features(features, batch, out_dir):
    def make_output_name(src, out_dir):
        basename = os.path.splitext(os.path.basename(src))[0]
        out_name = os.path.join(out_dir, '{}.npy'.format(basename))
        return out_name

    for feat, b in zip(features, batch):
        out_name = make_output_name(b, out_dir)
        print(out_name, feat.shape)
        np.save(out_name, feat)


def main(sess, args):
    module = hub.Module(args.module_url)
    height, width = hub.get_expected_image_size(module)
    print('Module: {}'.format(args.module_url))
    print('Height={} Width={}'.format(height, width))
    image_op = tf.placeholder(shape=(None, height, width, 3), 
        dtype=tf.float32)
    feat_op = module(image_op)
    sess.run(tf.global_variables_initializer())

    image_list = glob.glob(os.path.join(args.image_dir, '*.jpg'))
    n_images = len(image_list)
    n_batches = n_images // args.batch
    image_batches = np.array_split(image_list, n_batches)
    
    if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir)

    for batch in image_batches:
        images = load_batch(batch, height, width)
        features = sess.run(feat_op, {image_op: images})

        save_features(features, batch, args.out_dir)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_dir', type=str)
    parser.add_argument('--out_dir', type=str)
    parser.add_argument('--module_url', 
        type = str,
        default = 'https://tfhub.dev/google/imagenet/resnet_v2_101/feature_vector/1')
    parser.add_argument('--batch', default=12, type=int)

    args = parser.parse_args()

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    sess = tf.Session(config=config)
    main(sess, args)
    sess.close()
