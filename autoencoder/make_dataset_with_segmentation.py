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
import numpy as np
import argparse
import glob
import cv2
import sys
import os
import re

# sys.path.insert(0, '/Users/nathaning/_projects/tfmodels')
sys.path.insert(0, '/home/nathan/tfmodels')
import tfmodels

sys.path.insert(0, '.')
from nuclei_densenet_regression import DenseNetInference as regression
from nuclei_densenet_segmentation import DenseNetInference as segmentation
from normalize import reinhard

from scipy import ndimage as ndi
from skimage.feature import peak_local_max
from skimage.morphology import watershed


config = tf.ConfigProto()
config.gpu_options.allow_growth = True

segmentation_snapshot = 'segmentation_model/densenet.ckpt-5016'
regression_snapshot   = 'regression_model/densenet.ckpt-8319'

tiles_dir = 'sample_tiles'
window = 64
# for skimage.feature.peak_local_max
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

def get_img_nuclei(imgpath, reg_model):
    imgbase = os.path.basename(imgpath)
    img = read_image(imgpath)

    ## Normalization
    img_norm = reinhard(img)

    ## Regression mode
    img_reg = resize_image(img_norm, 512, 512)

    img_reg_scale = img_reg * (2/255.) - 1
    reg, _, _ = tfmodels.bayesian_inference(reg_model, np.expand_dims(img_reg_scale, 0), samples=10)
    reg = np.squeeze(reg)

    # Comparison between image_max and im to find the coordinates of local maxima
    try:
        reg_max = peak_local_max(reg, min_distance=min_distance, 
                                 threshold_abs=threshold_abs, indices=False)
        max_coord = peak_local_max(reg, min_distance=min_distance, 
                                   threshold_abs=threshold_abs, indices=True)
        wshed = instances(reg, reg_max)
    except:
        print(imgpath, 'Failed watershed')
        return None

    if np.random.binomial(1, 0.2):
        reg_max = cv2.dilate(reg_max.astype(np.uint8), 
            kernel= np.ones((3,3),np.uint8), iterations=1)
        img_out = overlay(img_reg, reg_max.astype(np.bool))
        cv2.imwrite(os.path.join(tiles_dir, imgbase), img_out)

    wshed = cv2.resize(wshed, dsize=(1000,1000), interpolation=cv2.INTER_NEAREST)
    nuclei = []
    for x, y in max_coord:
        x = int(x * ds_factor)
        y = int(y * ds_factor)
        label = wshed[x, y]
        bbox = get_box(x, y)
        if bbox is None:
            continue

        wsubimg = wshed[bbox[0]:bbox[1], bbox[2]:bbox[3]]
        subimg = img_norm[bbox[0]:bbox[1], bbox[2]:bbox[3], :]
        mask = wsubimg == label
        mask_area = mask.sum()
        if mask_area < area_threshold:
            continue

        mask = cv2.dilate(mask.astype(np.uint8), np.ones((5,5), np.uint8), iterations=3)

        subimg = apply_mask(subimg, mask)
        # subimg = correct_angle(subimg, mask)

        nuclei.append(np.expand_dims(subimg, axis=0))

    try:
        nuclei = np.concatenate(nuclei, axis=0)
    except:
        print('No nuclei found')
        return None

    return nuclei


def main():
    adeno_list = glob.glob('./adeno_tiles/*.tif')
    nepc_list = glob.glob( './nepc_tiles/*.tif')
    print('Images: {}'.format(len(adeno_list)))
    print('Images: {}'.format(len(nepc_list)))
    np.random.shuffle(adeno_list)
    np.random.shuffle(nepc_list)

    with tf.Session(config=config) as sess:
        model_args = {
            'sess': sess,
            'dense_stacks': [4, 4, 4],
            'growth_rate': 16,
            'x_dims': [512, 512, 3] }
        model = regression(**model_args)
        model.print_info()
        model.restore(regression_snapshot)

        all_nuclei = []
        for imgpath in adeno_list[:250]:
            img_nuclei = get_img_nuclei(imgpath, model)
            if img_nuclei is None:
                continue

            basename = os.path.splitext(os.path.basename(imgpath))[0]
            print('file {} containing {} nuclei'.format(basename, img_nuclei.shape))
            all_nuclei.append(img_nuclei)
        all_nuclei = np.concatenate(all_nuclei, axis=0)
        print('Adeno total: ', all_nuclei.shape)
        np.save('adeno_nuclei_images.npy', all_nuclei)

        all_nuclei = []
        for imgpath in nepc_list[:250]:
            img_nuclei = get_img_nuclei(imgpath, model)
            if img_nuclei is None:
                continue

            basename = os.path.splitext(os.path.basename(imgpath))[0]
            print('file {} containing {} nuclei'.format(basename, img_nuclei.shape))
            all_nuclei.append(img_nuclei)

        all_nuclei = np.concatenate(all_nuclei, axis=0)
        print('NEPC total: ', all_nuclei.shape)
        np.save('nepc_nuclei_images.npy', all_nuclei)

if __name__ == '__main__':
    main()
