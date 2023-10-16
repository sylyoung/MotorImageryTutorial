# -*- coding: utf-8 -*-
# @Time    : 2023/10/16
# @Author  : Siyang Li
# @File    : tsne.py
import mne
import numpy as np
import torch
from sklearn import preprocessing

import random

from utils.alg_utils import EA
from scipy.linalg import fractional_matrix_power

import matplotlib
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE


def data_process(dataset):
    '''
    :param dataset: str, dataset name
    :return: X, y, num_subjects, paradigm, sample_rate
    '''
    mne.set_log_level('warning')

    X = np.load('./data/' + dataset + '/X.npy')
    y = np.load('./data/' + dataset + '/labels.npy')
    print(X.shape, y.shape)

    num_subjects, paradigm, sample_rate = None, None, None

    if dataset == 'BNCI2014001':
        paradigm = 'MI'
        num_subjects = 9
        sample_rate = 250
        ch_num = 22

        # only use session T, remove session E
        indices = []
        for i in range(num_subjects):
            indices.append(np.arange(288) + (576 * i))
        indices = np.concatenate(indices, axis=0)
        X = X[indices]
        y = y[indices]

        # only use two classes [left_hand, right_hand]
        indices = []
        for i in range(len(y)):
            if y[i] in ['left_hand', 'right_hand']:
                indices.append(i)
        X = X[indices]
        y = y[indices]
    
    le = preprocessing.LabelEncoder()
    y = le.fit_transform(y)
    print('data shape:', X.shape, ' labels shape:', y.shape)
    return X, y, num_subjects, paradigm, sample_rate, ch_num


def data_alignment(X, num_subjects):
    '''
    :param X: np array, EEG data
    :param num_subjects: int, number of total subjects in X
    :return: np array, aligned EEG data
    '''
    # subject-wise EA
    #print('before EA:', X.shape)
    out = []
    for i in range(num_subjects):
        tmp_x = EA(X[X.shape[0] // num_subjects * i:X.shape[0] // num_subjects * (i + 1), :, :])
        out.append(tmp_x)
    X = np.concatenate(out, axis=0)
    #print('after EA:', X.shape)
    return X


def tsne_subjects(dataset):
    X, y, num_subjects, paradigm, sample_rate, ch_num = data_process(dataset)
    print('X, y, num_subjects, paradigm, sample_rate:', X.shape, y.shape, num_subjects, paradigm, sample_rate)

    print('sample rate:', sample_rate)

    # align
    #X = data_alignment(X, num_subjects)

    print('X, y', X.shape, y.shape)

    if paradigm == 'MI':
        # CSP
        csp = mne.decoding.CSP(n_components=10)

        X_csp = csp.fit_transform(X, y)

        tsne = TSNE(n_components=2, init='pca', random_state=0)
        features = tsne.fit_transform(X_csp)
        x_min, x_max = np.min(features, 0), np.max(features, 0)
        data = (features - x_min) / (x_max - x_min)

        csfont = {'fontname': "Times New Roman"}

        labels = []
        for i in range(num_subjects):
            labels.append(np.ones(len(y) // num_subjects) * i)  # assume equal num of trials across subjects
        labels = np.concatenate(labels)

        labels_colors = []
        markers = []

        labels_colors.append('red')
        labels_colors.append('blue')
        labels_colors.append('green')
        labels_colors.append('orange')
        labels_colors.append('purple')
        labels_colors.append('brown')
        labels_colors.append('pink')
        labels_colors.append('coral')
        labels_colors.append('grey')


        markers.append('^')
        markers.append('o')
        markers.append('x')
        markers.append('<')
        markers.append('>')
        markers.append('.')
        markers.append(',')
        markers.append('1')
        markers.append('2')


        X_2d = data

        for i in range(num_subjects):
            if len(markers) == 0:
                plt.scatter(X_2d[np.where(labels == i), 0], X_2d[np.where(labels == i), 1], s=3, c=labels_colors[i])
            else:
                plt.scatter(X_2d[np.where(labels == i), 0], X_2d[np.where(labels == i), 1], s=3, c=labels_colors[i], marker=markers[i])

        plt.savefig('./' + dataset + '_subjects.png', dpi=600, format='png')
        plt.savefig('./' + dataset + '_subjects.pdf', dpi=600, format='pdf')
        plt.savefig('./' + dataset + '_subjects.eps', dpi=600, format='eps')

        plt.clf()


if __name__ == '__main__':

    scores = []

    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    dataset = 'BNCI2014001'

    tsne_subjects(dataset)


