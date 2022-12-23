import random
import sys

import mne
import numpy as np
import torch
from mne.decoding import CSP
from sklearn import preprocessing
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier

from models.EEGNet import EEGNet
from nn_baseline import nn_fixepoch
from utils.alg_utils import EA
from utils.data_utils import traintest_split_cross_subject, dataset_to_file


def data_loader(dataset):
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
    print('before EA:', X.shape)
    out = []
    for i in range(num_subjects):
        tmp_x = EA(X[X.shape[0] // num_subjects * i:X.shape[0] // num_subjects * (i + 1), :, :])
        out.append(tmp_x)
    X = np.concatenate(out, axis=0)
    print('after EA:', X.shape)
    return X


def ml_classifier(approach, output_probability, train_x, train_y, test_x, return_model=None, weight=None):
    if approach == 'LDA':
        clf = LinearDiscriminantAnalysis()
    elif approach == 'LR':
        clf = LogisticRegression(max_iter=1000)
    elif approach == 'AdaBoost':
        clf = AdaBoostClassifier()
    elif approach == 'GradientBoosting':
        clf = GradientBoostingClassifier()
    elif approach == 'xgb':
        clf = XGBClassifier()
    clf.fit(train_x, train_y)
    pred = clf.predict(test_x)
    return pred


def sort_func_gen_data(name_string):
    id_ = -1
    if name_string.endswith('.npy'):
        id_ = int(name_string[-7:-4])
    return id_


def eeg_ml(dataset, info, align, approach, cuda_device_id):
    X, y, num_subjects, paradigm, sample_rate, ch_num = data_loader(dataset)
    # X, y, num_subjects, paradigm, sample_rate, ch_num = data_loader_feature(dataset)
    print('X, y, num_subjects, paradigm, sample_rate:', X.shape, y.shape, num_subjects, paradigm, sample_rate)

    if paradigm == 'ERP':
        print('ERP downsampled')
        X = mne.filter.resample(X, down=4)
        sample_rate = int(sample_rate // 4)
    print('sample rate:', sample_rate)

    if align:
        X = data_alignment(X, num_subjects)

    scores_arr = []

    for i in range(num_subjects):
        train_x, train_y, test_x, test_y = traintest_split_cross_subject(dataset, X, y, num_subjects, i)

        print('train_x, train_y, test_x, test_y.shape', train_x.shape, train_y.shape, test_x.shape, test_y.shape)

        if paradigm == 'MI':
            # CSP
            csp = CSP(n_components=10)
            train_x_csp = csp.fit_transform(train_x, train_y)
            test_x_csp = csp.transform(test_x)
            print('Training/Test split after CSP:', train_x_csp.shape, test_x_csp.shape)

            # classifier
            pred = ml_classifier(approach, False, train_x_csp, train_y, test_x_csp, return_model=True)
            score = np.round(accuracy_score(test_y, pred), 5)
            print('acc:', score)
        scores_arr.append(score)
    print('#' * 30)
    for i in range(len(scores_arr)):
        scores_arr[i] *= 100
        scores_arr[i] = np.round(scores_arr[i], 3)
    print('subject scores', scores_arr)
    print('avg', np.round(np.average(scores_arr), 3))
    print('#' * 30)

    return scores_arr


def eeg_dnn(dataset, info, align, approach, cuda_device_id):
    X, y, num_subjects, paradigm, sample_rate, ch_num = data_loader(dataset)
    print('sample rate:', sample_rate)

    if align:
        X = data_alignment(X, num_subjects)

    scores_arr = []
    for i in range(num_subjects):
        train_x, train_y, test_x, test_y = traintest_split_cross_subject(dataset, X, y, num_subjects, i)
        print('train_x, train_y, test_x, test_y.shape', train_x.shape, train_y.shape, test_x.shape, test_y.shape)
        class_out = len(np.unique(train_y))
        if paradigm == 'MI':
            metrics = accuracy_score
        # 5 random initializations
        seed_arr = np.arange(5)
        rand_init_scores = []
        for seed in seed_arr:
            model = EEGNet(n_classes=class_out,
                           Chans=train_x.shape[1],
                           Samples=train_x.shape[2],
                           kernLenght=int(sample_rate // 2),
                           F1=4,
                           D=2,
                           F2=8,
                           dropoutRate=0.25,
                           norm_rate=0.5)

            rand_init_score = nn_fixepoch(model=model,
                                          learning_rate=0.001,
                                          num_iterations=100,
                                          metrics=metrics,
                                          cuda=True,
                                          cuda_device_id=cuda_device_id,
                                          seed=int(seed),
                                          dataset=dataset,
                                          model_name='EEGNet',
                                          test_subj_id=i,
                                          label_probs=False,
                                          valid_percentage=0,
                                          train_x=train_x,
                                          train_y=train_y,
                                          test_x=test_x,
                                          test_y=test_y)

            rand_init_scores.append(rand_init_score)
        print('subj rand_init_scores:', rand_init_scores)
        score = np.round(np.average(rand_init_scores), 5)
        scores_arr.append(rand_init_scores)
        print('acc:', score)

    scores_arr = np.stack(scores_arr)
    print('#' * 40)
    scores_arr *= 100

    print('all scores', scores_arr)
    all_avgs = np.average(scores_arr, 0).round(3)
    print('all avgs', all_avgs)
    subj_stds = np.std(scores_arr, 0).round(3)
    print('sbj stds', subj_stds)
    all_avg = np.average(np.average(scores_arr, 0)).round(3)
    print('all avg', all_avg)
    all_std = np.std(np.average(scores_arr, 0)).round(3)
    print('all std', all_std)


if __name__ == '__main__':

    if len(sys.argv) == 1:
        cuda_device_id = 0
    else:
        cuda_device_id = str(sys.argv[1])
    try:
        device = torch.device('cuda:' + cuda_device_id)
        print('using GPU')
    except:
        device = torch.device('cpu')
        print('using CPU')

    scores = []

    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    dataset_arr = ['BNCI2014001']

    for dataset in dataset_arr:
        # Modify align to configure use Euclidean Alignment or not
        align = True

        # Modify datasave to False after download
        info = dataset_to_file(dataset, data_save=True)

        approach = 'LDA'
        print(dataset, align, approach)
        eeg_ml(dataset, info, align, approach, cuda_device_id)

        approach = 'EEGNet'
        print(dataset, align, approach)
        eeg_dnn(dataset, info, align, approach, cuda_device_id)
