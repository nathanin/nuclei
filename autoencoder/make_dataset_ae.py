"""
Save single case in one npz
1. Read image
2. Find nuclei
    a. Run two networks: Segmentation and distance
    b. Find local maxima in distnace
    c. Take positive area in window centered at each max point
        - window size = target_window * 2
    d. Get major axis of foreground area
    e. return rotated image s.t. major axis ~ vertical & backgorund is 0
3. Save Example with:
    'image': all rotated images as a stack
    'label'
    'case'
"""
import tensorflow as tf
import tensorflow.contrib.eager as tfe
import numpy as np
import pandas as pd
import argparse
import hashlib
import glob
import cv2
import sys
import os
import re

from scipy import ndimage as ndi
from skimage.feature import peak_local_max
from skimage.morphology import watershed

from autoencoder import Autoencoder

window = 64
min_distance = 8
threshold_abs = 4.0
area_threshold = 350
ds_factor = 1000 / 512.

def read_image(img_path):
    img = cv2.imread(img_path)
    return img

def resize_image(img, h, w):
    img = cv2.resize(img, dsize=(h, w))
    return img

def overlay(img, reg_max):
    r, g, b = np.split(img, 3, axis=-1)
    r[reg_max] = 50
    g[reg_max] = 240
    b[reg_max] = 50
    return np.dstack([r,g,b])

def get_box(x, y, max_x=1000, max_y=1000):
    x1 = int(x - window/2)
    x2 = int(x + window/2)
    y1 = int(y - window/2)
    y2 = int(y + window/2)

    if x1 < 0:
        return None

    if x2 > max_x:
        return None

    if y1 < 0:
        return None

    if y2 > max_y:
        return None

    return [x1, x2, y1, y2]

def instances(distance, maxes):
    mask = distance > 2
    markers = ndi.label(maxes, structure=np.ones((3,3)))[0]
    labels = watershed(-distance, markers, mask=mask)
    return labels

def apply_mask(img, mask):
    r, g, b = np.split(img, 3, axis=-1)
    r = np.squeeze(r); r = np.multiply(r, mask)
    g = np.squeeze(g); g = np.multiply(g, mask)
    b = np.squeeze(b); b = np.multiply(b, mask)
    return np.stack([r,g,b], axis=-1)

# def correct_angle(image, mask):
#     cnts,_ = cv2.findContours(mask.astype(np.uint8), 0, 0)
#     rows,cols = mask.shape
#     line = cv2.fitLine(cnts[0], cv.CV_DIST_L2, 0, 0.1, 0.1)
#     theta = np.rad2deg(np.arctan(line[1]/line[0])[0])*-1
# 
#     M = cv2.getRotationMatrix2D((cols/2,rows/2),theta,1)
#     image = cv2.warpAffine(image, M, (cols, rows))
#     return image

def read_indices(pth):
    try:
        with open(pth, 'r') as f:
            L = f.read()
    except:
        print('{} no lines'.format(pth))
        return None

    L = L.replace('\n', '')
    L = L.split(',')
    lout = []
    for x in L:
        try:
            lout.append(int(x))
        except:
            continue

    return lout

def get_img_nuclei(imgpath, maskpath, indexpath):
    imgbase = os.path.basename(imgpath)
    img = cv2.imread(imgpath)[:,:,::-1]

    mask = cv2.imread(maskpath, -1)

    indices = read_indices(indexpath)
    if indices is None:
        print('{}: No indices read'.format(indexpath))
        return None, None

    nuclei = []
    keep_indices = []
    for idx in indices:
        idx_mask = (mask == idx).astype(np.uint8)
        M = cv2.moments(idx_mask)
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])

        x1 = cX - int(window/2)
        x2 = cX + int(window/2)
        y1 = cY - int(window/2)
        y2 = cY + int(window/2)

        box = img[y1:y2, x1:x2, :]
        if box.shape[0] != window:
            print(imgpath, box.shape)
            continue
        if box.shape[1] != window:
            print(imgpath, box.shape)
            continue

        keep_indices.append(idx)
        nuclei.append(np.expand_dims(box, 0))

    try:
        nuclei = np.concatenate(nuclei, axis=0)
    except:
        print('{}: No nuclei found ({})'.format(indexpath, len(nuclei)))
        return None, None

    return nuclei, keep_indices

batchsize = 64
def main():
    data_head = pd.read_csv('../data/case_stage_files.tsv', 
                            sep='\t', index_col=0)
    print(data_head.head())

    model = Autoencoder()
    dummyx = tf.zeros((5, 64, 64, 3), dtype=tf.float32)
    _ = model(dummyx, verbose=True)
    saver = tfe.Saver(model.variables)
    saver.restore('./autoencoder_model/autoencoder-245000')
    model.summary()

    all_nuclei = []
    case_uids = []
    tile_uids = []
    nucleus_ids = []
    for k, tile_hash in enumerate(data_head.index.values):
        row = data_head.loc[tile_hash]
        tile_path = row['tile_path']
        mask_path = row['mask_path']
        index_path = row['index_path']
        case_id = row['case_id']

        tile_basename = os.path.basename(tile_path).replace('.tif', '')
        # Returns the successfully extracted nuclei and indices
        img_nuclei, indices = get_img_nuclei(tile_path, mask_path, index_path)
        if img_nuclei is None:
            continue

        case_uid = hashlib.md5(case_id.encode()).hexdigest() ## case_id is str
        n_nuclei = img_nuclei.shape[0]
        if n_nuclei <= batchsize:
            n_batches = 1
        else:
            n_batches = int(n_nuclei / batchsize)
        for xbatch, idx_batch in zip(np.array_split(img_nuclei, n_batches), 
                                     np.array_split(indices, n_batches)):

            xhat, zhat = model( tf.constant((xbatch / 255.).astype(np.float32)),
                                return_z=True )

            for ix in range(zhat.shape[0]):
            # for zhat_i, idx in zip(np.split(zhat, zhat.shape[0]),
            #                        np.split(idx_batch, zhat.shape[0])):
                    zhat_i = np.expand_dims(zhat[ix, :], 0)
                    idx = idx_batch[ix]
                    nuc_id = hashlib.md5('{}_{:04d}'.format(tile_basename, idx).encode()).hexdigest()
                    tile_id = hashlib.md5(tile_basename.encode()).hexdigest()
                    # print(tile_path, idx, nuc_id, case_uid, zhat_i.shape)

                    all_nuclei.append(zhat_i)
                    nucleus_ids.append(nuc_id)
                    case_uids.append(case_uid)
                    tile_uids.append(tile_id)

        if k % 250 == 0:
            print('{:05d}: {}\t{}\t{}\t{}'.format(k, tile_hash, 
                len(all_nuclei),
                len(nucleus_ids),
                len(case_uids),
                ))

    all_nuclei = np.concatenate(all_nuclei, axis=0)
    print('total: ', all_nuclei.shape)

    features = pd.DataFrame(all_nuclei, index=nucleus_ids)
    features['case_id'] = case_uids
    features['tile_id'] = tile_uids
    features.to_csv('../data/autoencoder_features.csv', sep=',')

if __name__ == '__main__':
    config = tf.ConfigProto()
    config.gpu_options.allow_growth =True
    tf.enable_eager_execution(config=config)
    main()
