# -------------------------------------------------------------------------------
# Name:        Inspect_Features.py
# Purpose:     Inspect Haar Features
# Usage:       python FeatureInspect.py
# Author:      Di Zhuang
# Created:     11/8/2015
# Copyright:   (c) Di Zhuang 2015
# -------------------------------------------------------------------------------


from FeatureExtract import load_feature_meta, pickle_load, HaarFeature
from FeatureTrain import TrainedFeatures
import cv2
import numpy as np
import argparse


def inspect_features(ls_features=range(162336), win_name='HaarFeatures'):
    features = load_feature_meta()

    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)

    # base_img = np.empty(shape=(24, 24), dtype=np.uint8)
    # base_img.fill(128)
    base_img = cv2.imread('face0002.jpg', 0)

    for feat in ls_features:
        stop_inspection = False

        img = np.copy(base_img)

        f = features[feat]
        print '{:s}'.format(f)

        height, width = f.type
        x, y = f.loc
        l, w = f.shape

        img[x : x + l, y : y + w] = 255

        # plot the feature
        if height == 2 and width == 1:
            img[x + l // 2 : x + l, y : y + w] = 0

        # if feature is type (1, 2),
        # feature value = sum(right rectangle) - sum(left rectangle)
        elif height == 1 and width == 2:
            img[x : x + l, y + w // 2 : y + w] = 0

        # if feature is type (3, 1),
        # feature value = sum(middle rectangle) - sum(top rectangle) - sum(bottom rectangle)
        elif height == 3 and width == 1:
            img[x + l // 3 : x + 2 * l // 3, y : y + w] = 0

        # if feature is type (1, 3),
        # feature value = sum of (middle rectangle) - sum (left rectangle) - sum(right rectangle)
        elif height == 1 and width == 3:
            img[x : x + l, y + w // 3 : y + 2 * w // 3] = 0

        # if feature is type (2, 2),
        # feature value = sum of (top right rectangle) + sum (bottom left rectangle)
        #                 - sum (top left rectangle) + sum(bottom right rectangle)
        elif height == 2 and width == 2:
            img[x : x + l // 2, y + w // 2 : y + w] = 0
            img[x + l // 2 : x + l, y : y + w // 2] = 0

        while True:
            key = cv2.waitKey(10)

            if key == 27:
                stop_inspection = True
                break
            if key == ord('n'):
                break

            cv2.imshow(win_name, img)

        if stop_inspection:
            break

    cv2.destroyAllWindows()


def inspect_best_features(best_features, win_name='HaarFeatures'):
    features = load_feature_meta()

    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)

    # base_img = np.empty(shape=(24, 24), dtype=np.uint8)
    # base_img.fill(128)

    base_img = cv2.imread('face0008.jpg', 0)

    for feat in best_features:
        min_feature_num, min_error, error_rate, threshold, parity, alpha = feat

        stop_inspection = False

        img = np.copy(base_img)

        f = features[min_feature_num]

        print
        print '{:s}'.format(f)
        print 'W. Error  = {:0.6f}'.format(min_error)
        print 'Error     = {:0.6f}'.format(error_rate)
        print 'Threshold = {:0.6f}'.format(threshold)
        print 'Parity      = {:0.6f}'.format(parity)
        print 'Alpha     = {:0.6f}'.format(alpha)

        height, width = f.type
        x, y = f.loc
        l, w = f.shape

        img[x : x + l, y : y + w] = 255

        # plot the feature
        if height == 2 and width == 1:
            img[x + l // 2 : x + l, y : y + w] = 0

        # if feature is type (1, 2),
        # feature value = sum(right rectangle) - sum(left rectangle)
        elif height == 1 and width == 2:
            img[x : x + l, y + w // 2 : y + w] = 0

        # if feature is type (3, 1),
        # feature value = sum(middle rectangle) - sum(top rectangle) - sum(bottom rectangle)
        elif height == 3 and width == 1:
            img[x + l // 3 : x + 2 * l // 3, y : y + w] = 0

        # if feature is type (1, 3),
        # feature value = sum of (middle rectangle) - sum (left rectangle) - sum(right rectangle)
        elif height == 1 and width == 3:
            img[x : x + l, y + w // 3 : y + 2 * w // 3] = 0

        # if feature is type (2, 2),
        # feature value = sum of (top right rectangle) + sum (bottom left rectangle)
        #                 - sum (top left rectangle) + sum(bottom right rectangle)
        elif height == 2 and width == 2:
            img[x : x + l // 2, y + w // 2 : y + w] = 0
            img[x + l // 2 : x + l, y : y + w // 2] = 0

        while True:
            key = cv2.waitKey(10)

            if key == 27:
                stop_inspection = True
                break
            if key == ord('n'):
                break

            cv2.imshow(win_name, img)

        if stop_inspection:
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Feature Inspection')
    parser.add_argument('-features', nargs='+', type=int, default=[66214],
                        help='inspect features by their id number')
    parser.add_argument('-all', action='store_true',
                        help='inspect all features chosen by Adaboost')

    args = parser.parse_args()

    if not args.all and len(args.features):
        inspect_features(args.features)  # inspect individual or a set of features given the feature id

    if args.all:
        best_features = pickle_load('adaboost10.pkl')
        # best_features = pickle_load('best_weak_learners_10.pkl')
        inspect_best_features(best_features)
