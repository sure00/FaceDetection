# -------------------------------------------------------------------------------
# Name:        FeatureExtract.py
# Purpose:     Implement Viola_Jones
# Usage:       python Viola_Jones.py
# Author:      Di Zhuang
# Created:     10/27/2015
# Copyright:   (c) Di Zhuang 2015
# -------------------------------------------------------------------------------

import numpy as np
import cv2
import os
import pickle
import time
import warnings
import argparse
from collections import namedtuple


DEBUG = False

"""
A Haar Feature is a named tuple (similar to struct in C) and it has 3 fields:
    type:   a 2-tuple that specifies the type of Haar Feature.  It can be one of the following:
            (2, 1), (1, 2), (3, 1), (1, 3), (2, 2)
    loc:    a 2-tuple that specifies the location of this Haar Feature.
            (x, y) means that this feature starts on row x and column y
    shape:  a 2-tuple that specifies the shape of this feature.
            (x, y) means that the feature spans x rows and y columns in the original frame
"""
HaarFeature = namedtuple('HaarFeature', ['type', 'loc', 'shape'])


"""
Types of haar features to be used in Viola-Jones Algorithm
"""
HAAR_FEATURE_TYPES = [(2, 1), (1, 2), (3, 1), (1, 3), (2, 2)]


def feature_extract(img):
    """
    Extract features from training images

    :param img: source image (24x24 pixeled images)
    :return: a vector of all features computed from source image
             (for the default frame size of 24x24, this returns 162,336 features)
    """

    # compute the integral image of the source image
    normalized_src = normalize(img)
    iimg = integral_image(normalized_src)

    row, col = iimg.shape

    features = []

    # for each feature
    for feature in HAAR_FEATURE_TYPES:
        height, width = feature

        # for each feature size (x, y)
        for x in xrange(height, row, height):
            for y in xrange(width, col, width):
                # for each starting position
                for i in xrange(row-x):
                    for j in xrange(col-y):
                        # if feature is type (2, 1),
                        # feature value = sum(top rectangle) - sum(bottom rectangle)
                        if height == 2 and width == 1:
                            f = iimg[i, j] - iimg[i, j+y] - 2 * iimg[i+x//2, j] \
                                + 2 * iimg[i+x//2, j+y] + iimg[i+x, j] - iimg[i+x, j+y]

                        # if feature is type (1, 2),
                        # feature value = sum(right rectangle) - sum(left rectangle)
                        elif height == 1 and width == 2:
                            f = -iimg[i, j] + 2 * iimg[i, j+y//2] - iimg[i, j+y] \
                                + iimg[i+x, j] - 2 * iimg[i+x, j+y//2] + iimg[i+x, j+y]

                        # if feature is type (3, 1),
                        # feature value = sum(middle rectangle) - sum(top rectangle) - sum(bottom rectangle)
                        elif height == 3 and width == 1:
                            f = -iimg[i, j] + iimg[i, j+y] + 2 * iimg[i+x//3, j] \
                                - 2 * iimg[i+x//3, j+y] - 2 * iimg[i+2*x//3, j] + 2 * iimg[i+2*x//3, j+y] \
                                + iimg[i+x, j] - iimg[i+x, j+y]

                        # if feature is type (1, 3),
                        # feature value = sum of (middle rectangle) - sum (left rectangle) - sum(right rectangle)
                        elif height == 1 and width == 3:
                            f = -iimg[i, j] + 2 * iimg[i, j+y//3] - 2 * iimg[i, j+2*y//3] \
                                + iimg[i, j+y] + iimg[i+x, j] - 2 * iimg[i+x, j+y//3] \
                                + 2 * iimg[i+x, j + 2*y//3] - iimg[i+x, j+y]

                        # if feature is type (2, 2),
                        # feature value = sum of (top right rectangle) + sum (bottom left rectangle)
                        #                 - sum (top left rectangle) + sum(bottom right rectangle)
                        elif height == 2 and width == 2:
                            f = -iimg[i, j] + 2 * iimg[i, j+y//2] - iimg[i, j+y] \
                                + 2 * iimg[i+x//2, j] - 4 * iimg[i+x//2, j+y//2] + 2 * iimg[i+x//2, j+y] \
                                - iimg[i+x, j] + 2 * iimg[i+x, j+y//2] - iimg[i+x, j+y]

                        features.append(f)

    return features


def normalize(img):
    dst = np.copy(img.astype(np.float32))
    dst = (dst - np.mean(dst)) / np.std(dst)
    return dst


def integral_image(img):
    iimg = np.vstack((np.zeros(img.shape[1], dtype=np.float32), img))
    iimg = np.hstack((np.zeros((img.shape[0]+1, 1), dtype=np.float32), iimg))

    return iimg.cumsum(1).cumsum(0)


def feature_meta(frame_size=24, pickle_file='haar_feature_info.pkl'):
    """
    Generate the haar features meta-information which allow inspection into
    which Haar features are deemed relevant according to Adaboost and Cascade Classifier.
    The list of feature meta information are saved into a pickle file.

    :param frame_size: default size of each frame to extract Haar features from
    :param pickle_file: name of the file to save the meta information into
    :return: a list of HaarFeature objects which contains the relevant information of each
             feature
    """
    row, col = frame_size + 1, frame_size + 1

    features = []

    # for each feature
    for feature in HAAR_FEATURE_TYPES:
        height, width = feature

        # for each feature size
        for x in xrange(height, row, height):
            for y in xrange(width, col, width):
                # for each starting position
                for i in xrange(row-x):
                    for j in xrange(col-y):
                        features.append(HaarFeature(type=(height, width), loc=(i, j), shape=(x, y)))

    pickle_save(pickle_file, features)

    return features


def load_feature_meta(filename='haar_feature_info.pkl'):
    """
    Read the feature meta information

    :param filename: name of the haar feature file
    :return:
    """
    return pickle_load(filename)


def extract_feature_from_images(face_path='./faceTrainingData/faces', nonface_path='./faceTrainingData/nonfaces',
                                save_dir='./features_test/'):
    """
    Extract features from training set

    :param face_path: directory containing the face training images
    :param nonface_path: directory containing the non-face training images
    :param save_dir: directory to save all the features
    :return:
    """

    start = time.time()

    print 'Training ...'

    features = []
    labels = []

    parent_dir = os.getcwd().replace('\\', '/')

    os.chdir(face_path)
    faces = len(os.listdir('.'))

    for img_id, face_img in enumerate(os.listdir('.')):
        if img_id + 1 < 4100:
            continue

        img = cv2.imread(face_img, 0)
        features.append(feature_extract(img))
        labels.append(1)
        if (img_id + 1) % 100 == 0 or img_id == faces - 1:
            filename = parent_dir + save_dir + 'faces{:04d}_{:04d}.pkl'.format(img_id // 100 * 100 + 1, img_id + 1)
            print 'Saving features from images {:04d} - {:04d} in {:s}'.format(
                    img_id // 100 * 100 + 1, img_id + 1, filename)

            pickle_save(filename,
                        (np.array(features, dtype=np.float32),
                        np.array(labels, dtype=np.float32)))

            features = []  # reset the number of features
            labels = []  # reset the number of labels

    os.chdir(parent_dir)

    os.chdir(nonface_path)
    non_faces = len(os.listdir('.'))

    for img_id, nonface_img in enumerate(os.listdir('.')):
        img = cv2.imread(nonface_img, 0)
        features.append(feature_extract(img))
        labels.append(0)
        if (img_id + 1) % 100 == 0 or img_id == non_faces - 1:
            filename = parent_dir + save_dir + 'nonfaces{:04d}_{:04d}.pkl'.format(img_id // 100 * 100 + 1, img_id + 1)
            print 'Saving features from images {:04d} - {:04d} in {:s}'.format(
                    img_id // 100 * 100 + 1, img_id + 1, filename)

            pickle_save(filename,
                        (np.array(features, dtype=np.float32),
                        np.array(labels, dtype=np.float32)))

            features = []  # reset the number of features
            labels = []  # reset the number of labels

    os.chdir(parent_dir)

    print 'Training finished in %5.2f secs.' % (time.time() - start)
    print 'There are %d images in the training set (%d faces and %d non-faces)' % \
          (faces + non_faces, faces, non_faces)


def pickle_save(filename, obj):
    """
    save object to file using pickle

    :param filename: filename to save the object in
    :param obj: object
    :return:
    """
    with open(filename, 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


def pickle_load(filename):
    """
    load object from pickle file

    :param filename: name of the file
    :return:
    """
    with open(filename, 'rb') as f:
        obj = pickle.load(f)

    return obj


def read_features(filename, f_start, f_end):
    """
    Read feature from file

    :param filename: feature filename
    :param f_start: the start of the feature num
    :param f_end: the end of the feature num
    :return:
    """
    with open(filename, 'rb') as f:
        features, labels = pickle.load(f)

    return features[:, f_start:f_end], labels


def split_features(data_path, save_path, feature_total=162336, chunks=4):
    """
    Split the feature data into even sized chunks by feature num.
    For example,
    feature 1 to 40584 will be written to features0001_40584.pkl

    :param data_path: path of features extracted from images
    :param save_path: path of to save the new features
    :param feature_total: total number of features
    :param chunks: number of even sized chunks to make
    :return:
    """
    parent = os.getcwd()

    size = feature_total // chunks

    for feature_start in xrange(0, feature_total, size):
        os.chdir(data_path)
        all_features = None
        all_labels = None
        for i, pkl_file in enumerate(sorted(os.listdir('.'))):
            print 'reading %s ...' % pkl_file
            feature, labels = read_features(pkl_file, feature_start, feature_start + size)

            if i == 0:
                all_features = feature
                all_labels = labels
            else:
                all_features = np.vstack((all_features, feature))
                all_labels = np.hstack((all_labels, labels))

        filename = 'features{:04d}_{:04d}.pkl'.format(feature_start + 1, feature_start + size)

        os.chdir(parent)

        os.chdir(save_path)
        print 'saving features to %s ...' % filename
        pickle_save(filename, (all_features, all_labels))
        print 'saving complete!'

        os.chdir(parent)


if __name__ == "__main__":
    warnings.simplefilter("ignore")

    parser = argparse.ArgumentParser(description='Feature Extraction')
    parser.add_argument('-extract', action='store_true', help='extract features from training images')
    parser.add_argument('-split', action='store_true', help='split the feature data by feature num')

    args = parser.parse_args()

    if args.extract:
        # extract features from all training images
        extract_feature_from_images(save_dir='/image_features/')

    if args.split:
        # split the features into even sized chunks
        split_features(data_path='./image_features/', save_path='./features/')
