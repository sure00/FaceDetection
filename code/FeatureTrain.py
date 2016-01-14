# -------------------------------------------------------------------------------
# Name:        FeatureTrain.py
# Purpose:     Train strong features using Viola Jones Algorithm
# Usage:       python FeatureTrain.py
# Author:      Di Zhuang
# Created:     11/13/2015
# Copyright:   (c) Di Zhuang 2015
# -------------------------------------------------------------------------------

from __future__ import division
import numpy as np
import os
import pickle
import time
import warnings
import argparse
from collections import namedtuple
from FeatureExtract import pickle_save, pickle_load
from sklearn.linear_model import LogisticRegression


DEBUG = False


"""
Trained Feature is a named tuple (similar to struct in C) and it has 6 fields:
    feature_num:  id of the feature. can use HaarFeature meta to find its type, location, and shape
    weighted_err: weighted error value when this classifer was selected
    error:        the error rate of this classifier when this feature was selected
    threshold:    y = 1 when parity * feature > parity * threshold
                  y = 0 otherwise
    parity:       either -1 or 1 (it changes the sign of the inequality above)
    alpha:        -log(beta)
"""
TrainedFeatures = namedtuple('TrainedFeatures',
                             ['feature_num', 'weighted_error', 'error', 'threshold', 'parity', 'alpha'])


def feature_train(data_path, weak_learner_file='all_weak_learners.pkl',
                  features_file='predict_error.pkl', num_pos=4916, num_neg=7960, n_estimators=200):

    start = time.time()

    print 'Starting Adaboost training ...'

    parent_dir = os.getcwd()
    os.chdir(data_path)

    weak_learners = pickle_load(weak_learner_file)
    features = pickle_load(features_file)

    os.chdir(parent_dir)

    # setup the initial weights
    data_weights = np.empty(num_pos+num_neg, dtype=np.float32)
    data_weights[:num_pos] = 1 / 2 / num_pos
    data_weights[num_pos:] = 1 / 2 / num_neg
    labels = np.zeros(num_pos+num_neg, dtype=np.int8)
    labels[:num_pos] = 1

    best_weak_learners = []

    for i in xrange(n_estimators):
        # find the best weak learner by computing the minimum error
        round_start = time.time()

        error = np.dot(features, data_weights)

        # select the best weak learner
        min_feature_num = np.argmin(error)
        min_error = error[min_feature_num]
        pred_errs = features[min_feature_num]
        error_rate, threshold, parity = weak_learners[min_feature_num]

        # update the data weights
        beta = min_error / (1 - min_error)
        alpha = -np.log(beta)
        Beta = np.copy(data_weights)
        Beta.fill(beta)
        data_weights = data_weights * np.power(Beta, np.logical_not(pred_errs))
        data_weights = data_weights / np.sum(data_weights)  # normalize the weights
        hist, bin_edges = np.histogram(data_weights, bins=10, range=(0, 1), density=True)
        print hist

        # ensure the feature selected will NOT be selected again
        # this ensures that the previously selected feature have the worst error rates
        features[min_feature_num] = 2

        print 'Iter #{:3d}'.format(i+1)
        print '-' * 30
        print 'Feature   = {: 06d}'.format(min_feature_num)
        print 'W. Error  = {:0.6f}'.format(min_error)
        print 'Error     = {:0.6f}'.format(error_rate)
        print 'Threshold = {:0.6f}'.format(threshold)
        print 'Beta      = {:0.6f}'.format(beta)
        print 'Alpha     = {:0.6f}'.format(alpha)
        print 'This round took %5.2f secs.' % (time.time() - round_start)
        print '-' * 30
        print

        # save this weak classifier
        best_weak_learners.append(TrainedFeatures(min_feature_num, min_error, error_rate,
                                                  threshold, parity, alpha))

    print 'Finished Adaboost training in %5.2f secs.' % (time.time() - start)

    return best_weak_learners


def read_features(filename):
    print 'reading features from %s' % filename

    with open(filename, 'rb') as f:
        features, labels = pickle.load(f)

    return features, labels


def build_weak_learners(data_path='./features', num_features=162336, chunks=4,
                        num_pos=4916, num_neg=7960):

    start = time.time()

    print 'Building weak learners ...'

    parent_dir = os.getcwd()

    os.chdir(data_path)

    size = num_features // chunks

    logit = LogisticRegression()

    # find the best weak learner (logistic regression)
    for j in xrange(0, num_features, size):
        features, labels = read_features(filename='features{:04d}_{:04d}.pkl'.format(j + 1, j + size))

        # weak_learners = np.empty(shape=(size, 3), dtype=np.float32)
        # pred_features = np.empty((size, num_pos+num_neg), dtype=np.int8)

        for offset in xrange(size):
            feature_num = j + offset

            feature = features[:, offset]
            f = feature.reshape(features.shape[0], 1)
            logit.fit(f, labels)
            threshold = -1 * logit.intercept_[0] / logit.coef_[0, 0]
            parity = 1 if logit.coef_ >= 0 else -1

            pred = logit.predict(f)
            error = np.count_nonzero(labels != pred) / len(labels)

            print 'Feature #{:06d} | Error = {:0.6f} | Threshold = {: 15.6f} | parity = {:1d}'\
                .format(feature_num, error, threshold, parity)

            # save this weak classifier
            # weak_learners[feature_num] = np.array([error, threshold, parity], dtype=np.float32)
            # pred_features[feature_num, idx] = 1

        # pickle_save('weak_learners{:05d}_{:05d}.pkl'.format(j + 1, j + size), weak_learners)
        # pickle_save('all_features{:05d}_{:05d}.pkl'.format(j + 1, j + size), pred_features)

    os.chdir(parent_dir)

    print 'Finished training all weak learners in %5.2f secs.' % (time.time() - start)

    return
    # return weak_learners, pred_features


def preprocess():
    build_weak_learners()


def train(data_path='./', num_features=200):
    clf = feature_train(data_path, n_estimators=num_features)
    pickle_save('adaboost{:d}.pkl'.format(num_features), clf)
    return clf


def analyze_features(best_features='adaboost300.pkl', features_file='pred_features.pkl',
                     num_pos=4916, num_neg=7960):
    from sklearn.metrics import confusion_matrix

    best_features = pickle_load(best_features)
    features = pickle_load(features_file)

    sum_of_alphas = 0
    n_estimators = len(best_features)

    labels = np.zeros(num_pos+num_neg, dtype=np.int8)
    labels[:num_pos] = 1

    accuracy_rates = np.empty(n_estimators, dtype=np.float32)
    false_positive_rates = np.empty(n_estimators, dtype=np.float32)

    for i, estimator in enumerate(best_features):
        min_feature_num, min_error, error_rate, threshold, parity, alpha = estimator

        sum_of_alphas += alpha

        if i == 0:
            combined_threshold = features[min_feature_num] * alpha
        else:
            combined_threshold += features[min_feature_num] * alpha

        pred = combined_threshold > sum_of_alphas / 2

        cmat = confusion_matrix(labels, pred)

        # compute the false positive of the current final classifier
        # (using all features selected by Adaboost up to i+1)
        false_positive_rates[i] = cmat[1, 0] / (cmat[0, 0] + cmat[1, 0])

        # compute the accuracy of the final classifier
        # (using all features selected by Adaboost up to i+1)
        accuracy_rates[i] = (cmat[0, 0] + cmat[1, 1]) / len(pred)

    return accuracy_rates, false_positive_rates


if __name__ == '__main__':
    warnings.simplefilter("ignore")

    parser = argparse.ArgumentParser(description='Feature Extraction')
    parser.add_argument('-preprocess', action='store_true',
                        help='preprocess the features by training weak learners from them')
    parser.add_argument('-train', type=int, default=500,
                        help='train a final classifier with N features using Adaboost')

    args = parser.parse_args()

    if args.preprocess:
        preprocess()

    if args.train:
        best_features = train(num_features=args.train)
