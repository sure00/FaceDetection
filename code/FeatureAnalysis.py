# -------------------------------------------------------------------------------
# Name:        FeatureTrain.py
# Purpose:     Train strong features using Viola Jones Algorithm
# Usage:       python FeatureTrain.py
# Author:      Di Zhuang
# Created:     11/13/2015
# Copyright:   (c) Di Zhuang 2015
# -------------------------------------------------------------------------------

from __future__ import division

import warnings
import numpy as np
from FeatureExtract import pickle_load
from FeatureTrain import TrainedFeatures
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt


def analyze_features(best_features='best_weak_learners_200.pkl', features_file='all_predict_features.pkl',
                     num_pos=4916, num_neg=7960, top=None):
    """
    :param best_features: file containing a list of the best classifiers selected by adaboost
    :param features_file: file containing a matrix of predicted features
    :param num_pos: number of positive images in the training set
    :param num_neg: number of negative images in the training set
    :return:
    """
    best_features = pickle_load(best_features)
    predict_features = pickle_load(features_file)

    sum_of_alphas = 0
    n_estimators = len(best_features)

    labels = np.zeros(num_pos+num_neg, dtype=np.int8)
    labels[:num_pos] = 1

    if top is None:
        top = n_estimators

    accuracy_rates = np.empty(top, dtype=np.float32)
    false_positive_rates = np.empty(top, dtype=np.float32)

    for i, estimator in enumerate(best_features[:top]):
        min_feature_num, min_error, error_rate, threshold, parity , alpha = estimator

        sum_of_alphas += alpha
        pred_features = np.ravel(predict_features[min_feature_num].todense())

        if i == 0:
            combined_threshold = pred_features * alpha
        else:
            combined_threshold += pred_features * alpha

        pred = combined_threshold > sum_of_alphas / 2

        cmat = confusion_matrix(labels, pred)

        # compute the false positive of the current final classifier
        # (using all features selected by Adaboost up to i+1)
        false_positive_rates[i] = cmat[1, 0] / (cmat[0, 0] + cmat[1, 0])

        # compute the accuracy of the final classifier
        # (using all features selected by Adaboost up to i+1)
        accuracy_rates[i] = (cmat[0, 0] + cmat[1, 1]) / len(pred)

    return accuracy_rates, false_positive_rates


def plot(accuracy_scores, false_positive_scores):
    """
    Plot a graph showing the classifier performance progression.

    :param accuracy_scores:  accuracy rate
    :param false_positive_scores:  false positive rate
    :return:
    """
    axes_params = [0.1, 0.1, 0.58, 0.75]
    bbox_anchor_coord = (1.02, 1)

    # Plot the results
    fig = plt.figure(1)
    ax = fig.add_axes(axes_params)

    classifiers = np.arange(1, len(accuracy_scores) + 1)
    ax.plot(classifiers, accuracy_scores, label='Accuracy Rate', color='r', ls=':')
    ax.plot(classifiers, false_positive_scores, label='False-Positive Rate', color='b', ls='-.')
    ax.legend(bbox_to_anchor=bbox_anchor_coord, loc=2)
    plt.xlabel('# of classifiers')
    plt.ylabel('Rate')
    plt.xlim(1, len(accuracy_scores)+1)
    plt.ylim(0, 1)
    plt.title('Performance of Viola-Jones Adaboosted Classifier')
    plt.show(block=True)


if __name__ == '__main__':
    warnings.simplefilter("ignore")
    acc_rates, fp_rates = analyze_features(best_features='adaboost10.pkl', top=None)
    plot(acc_rates, fp_rates)

