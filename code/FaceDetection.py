# -------------------------------------------------------------------------------
# Name:        FaceDetection.py
# Purpose:     Implement face detection using Python and OpenCV
# Usage:       python FaceDetection.py
# Author:      Di Zhuang
# Created:     11/13/2015
# Copyright:   (c) Di Zhuang 2015
# -------------------------------------------------------------------------------

from __future__ import division

import argparse
import warnings
from collections import namedtuple

import cv2
import numpy as np

from FeatureExtract import load_feature_meta, pickle_load, pickle_save, normalize
from FeatureExtract import HaarFeature
from FeatureTrain import TrainedFeatures

FaceDetector = namedtuple('FaceDetector', {'feature_num', 'threshold', 'parity', 'alpha', 'type', 'loc', 'shape'})


def face_detect(src, frame=(24, 24), scale=1.2):
    frame_height, frame_width = frame

    faces = []

    current_scale = 1

    dst = normalize(src)
    img = np.copy(dst)

    face_detector = load_face_detector()

    while img.shape[0] >= frame_height and img.shape[1] >= frame_width:
        alphas = []
        features = []

        # for each starting location
        for row in xrange(img.shape[0] - frame_height + 1):
            for col in xrange(img.shape[1] - frame_width + 1):
                iimg = cv2.integral(img[row:row+frame_height, col:col+frame_width])

                for learner in face_detector:
                    min_feature_num, threshold, parity, alpha, f_type, f_loc, f_shape = learner
                    alphas.append(alpha)

                    height, width = f_type
                    i, j = f_loc
                    x, y = f_shape

                    # compute feature value

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

                    features.append(int(f * parity > threshold * parity))

                weighted_sum = np.dot(np.array(features), np.array(alphas))
                params = (int(row * current_scale), int(col * current_scale),
                          int(frame_height * current_scale), int(frame_width * current_scale))

                if weighted_sum > np.sum(alphas) / 2:
                    # there is a frame in this frame
                    faces.append(params)
                    print 'face found at {}'.format(faces[-1])
                else:
                    print 'face not found at {}'.format(params)

        img = cv2.resize(img, (int(img.shape[0] / scale), int(img.shape[1] / scale)))
        current_scale *= scale

    return faces


def resolve_overlaps(src, faces, window_sz=20, threshold=1):
    # handles overlapping windows
    # strategy: sum up all the windows with faces in them
    # put a sliding window on the face_counter image and select the largest ones
    face_counter = np.zeros(src.shape, dtype=np.int8)
    area_selected = np.zeros(src.shape, dtype=np.int8)

    selected_faces = []

    for face in faces:
        row, col, x, y = face
        face_counter[row:row+x, col:col+y] += 1

    for i in xrange(src.shape[0]-window_sz):
        for j in xrange(src.shape[1]-window_sz):
            if face_counter[i, j] > threshold and np.any(area_selected[i:i+window_sz, j:j+window_sz]):
                selected_faces.append((i, j, window_sz, window_sz))
                area_selected[i:i+window_sz, j:j+window_sz] = 1

    return selected_faces


def build_face_detector(filename='face_detector.pkl', best_learner_file='adaboost500.pkl', top=200):
    best_learners = pickle_load(best_learner_file)
    feature_info = load_feature_meta()

    face_detector = []
    for learner in best_learners[:top]:
        min_feature_num, min_error, error_rate, threshold, parity, alpha = learner
        f = feature_info[min_feature_num]
        face_detector += [FaceDetector(min_feature_num, threshold, parity, alpha, f.type, f.loc, f.shape)]

    pickle_save(filename, face_detector)


def load_face_detector(filename='face_detector.pkl'):
    return pickle_load(filename)


if __name__ == '__main__':
    warnings.simplefilter("ignore")

    parser = argparse.ArgumentParser(description='FaceDetection')
    parser.add_argument('-i', type=str, default='face0008.jpg',
                        help='run face detection on image using default classifier')
    parser.add_argument('-l', type=str, default='adaboost10.pkl',
                        help='run face detection on image using customized classifier')

    args = parser.parse_args()

    if args.l != 'adaboost10.pkl':
        build_face_detector(best_learner_file=args.l, top=None)
    else:
        gray = cv2.imread(args.i, 0)
        print face_detect(gray)
