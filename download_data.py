import os

import numpy as np
import moabb

from moabb.datasets import BNCI2014001, BNCI2014002, BNCI2014004, BNCI2015001, Cho2017, PhysionetMI
from moabb.paradigms import MotorImagery, P300


def dataset_to_file(dataset_name, data_save):
    moabb.set_log_level("ERROR")
    if dataset_name == 'BNCI2014001':
        dataset = BNCI2014001()
        paradigm = MotorImagery(n_classes=4)
        # (5184, 22, 1001) (5184,) 250Hz 9subjects * 4classes * (72+72)trials for 2sessions
    elif dataset_name == 'BNCI2014002':
        dataset = BNCI2014002()
        paradigm = MotorImagery(n_classes=2)
        # (2240, 15, 2561) (2240,) 512Hz 14subjects * 2classes * (50+30)trials * 2sessions(not namely separately)
    elif dataset_name == 'BNCI2014004':
        dataset = BNCI2014004()
        paradigm = MotorImagery(n_classes=2)
        # (6520, 3, 1126) (6520,) 250Hz 9subjects * 2classes * (?)trials * 5sessions
    elif dataset_name == 'BNCI2015001':
        dataset = BNCI2015001()
        paradigm = MotorImagery(n_classes=2)
        # (5600, 13, 2561) (5600,) 512Hz 12subjects * 2 classes * (200 + 200 + (200 for Subj 8/9/10/11)) trials * (2/3)sessions
    elif dataset_name == 'Cho2017':
        dataset = Cho2017()
        paradigm = MotorImagery(n_classes=2)
        #
    elif dataset_name == 'PhysionetMI':
        dataset = PhysionetMI(imagined=True, executed=False)
        paradigm = MotorImagery(n_classes=2)
        #

    if data_save:
        print('preparing ' + str(dataset_name) + ' data...')
        X, labels, meta = paradigm.get_data(dataset=dataset, subjects=dataset.subject_list[:])
        ar_unique, cnts = np.unique(labels, return_counts=True)
        print("labels:", ar_unique)
        print("Counts:", cnts)
        print(X.shape, labels.shape)
        if not os.path.exists('./data/'):
            os.makedirs('./data/')
        if not os.path.exists('./data/' + dataset_name + '/'):
            os.makedirs('./data/' + dataset_name + '/')
        np.save('./data/' + dataset_name + '/X', X)
        np.save('./data/' + dataset_name + '/labels', labels)
        meta.to_csv('./data/' + dataset_name + '/meta.csv')
        print('done!')
    else:
        if isinstance(paradigm, MotorImagery):
            X, labels, meta = paradigm.get_data(dataset=dataset, subjects=[dataset.subject_list[0]], return_epochs=True)
            return X.info
        elif isinstance(paradigm, P300):
            X, labels, meta = paradigm.get_data(dataset=dataset, subjects=[dataset.subject_list[0]], return_epochs=True)
            return X.info


if __name__ == '__main__':


    #datasets = ['BNCI2014001', 'BNCI2014002', 'BNCI2014004', 'BNCI2015001']
    datasets = ['BNCI2014001']
    for dataset_name in datasets:
        info = dataset_to_file(dataset_name, data_save=True)


    '''
    BNCI2014001
    <Info | 8 non-empty values
     bads: []
     ch_names: 'Fz', 'FC3', 'FC1', 'FCz', 'FC2', 'FC4', 'C5', 'C3', 'C1', 'Cz', 'C2',
      'C4', 'C6', 'CP3', 'CP1', 'CPz', 'CP2', 'CP4', 'P1', 'Pz', 'P2', 'POz'
     chs: 22 EEG
     custom_ref_applied: False
     dig: 25 items (3 Cardinal, 22 EEG)
     highpass: 8.0 Hz
     lowpass: 32.0 Hz
     meas_date: unspecified
     nchan: 22
     projs: []
     sfreq: 250.0 Hz
    >

    BNCI2014002
    <Info | 7 non-empty values
     bads: []
     ch_names: 'EEG1', 'EEG2', 'EEG3', 'EEG4', 'EEG5', 'EEG6', 'EEG7', 'EEG8', 'EEG9', 'EEG10', 'EEG11', 'EEG12', 'EEG13', 'EEG14', 'EEG15'
     chs: 15 EEG
     custom_ref_applied: False
     highpass: 8.0 Hz
     lowpass: 32.0 Hz
     meas_date: unspecified
     nchan: 15
     projs: []
     sfreq: 512.0 Hz
    >
    
    BNCI2014004
    <Info | 8 non-empty values
     bads: []
     ch_names: 'C3', 'Cz', 'C4'
     chs: 3 EEG
     custom_ref_applied: False
     dig: 6 items (3 Cardinal, 3 EEG)
     highpass: 8.0 Hz
     lowpass: 32.0 Hz
     meas_date: unspecified
     nchan: 3
     projs: []
     sfreq: 250.0 Hz
    >

    BNCI2015001
    <Info | 8 non-empty values
     bads: []
     ch_names: 'FC3', 'FCz', 'FC4', 'C5', 'C3', 'C1', 'Cz', 'C2', 'C4', 'C6', 'CP3', 'CPz', 'CP4'
     chs: 13 EEG
     custom_ref_applied: False
     dig: 16 items (3 Cardinal, 13 EEG)
     highpass: 8.0 Hz
     lowpass: 32.0 Hz
     meas_date: unspecified
     nchan: 13
     projs: []
     sfreq: 512.0 Hz
    >
    
    Cho2017
    <Info | 8 non-empty values
     bads: []
     ch_names: 'Fp1', 'AF7', 'AF3', 'F1', 'F3', 'F5', 'F7', 'FT7', 'FC5', 'FC3', 'FC1', 'C1', 'C3', 'C5', 'T7', 'TP7', 'CP5', 'CP3', 'CP1', 'P1', 'P3', 'P5', 'P7', 'P9', 'PO7', 'PO3', 'O1', 'Iz', 'Oz', 'POz', 'Pz', 'CPz', 'Fpz', 'Fp2', 'AF8', 'AF4', 'AFz', 'Fz', 'F2', 'F4', 'F6', 'F8', 'FT8', 'FC6', 'FC4', 'FC2', 'FCz', 'Cz', 'C2', 'C4', 'C6', 'T8', 'TP8', 'CP6', 'CP4', 'CP2', 'P2', 'P4', 'P6', 'P8', 'P10', 'PO8', 'PO4', 'O2'
     chs: 64 EEG
     custom_ref_applied: False
     dig: 67 items (3 Cardinal, 64 EEG)
     highpass: 8.0 Hz
     lowpass: 32.0 Hz
     meas_date: unspecified
     nchan: 64
     projs: []
     sfreq: 512.0 Hz
    >
    
    PhysionetMI
    <Info | 8 non-empty values
     bads: []
     ch_names: ['FC5', 'FC3', 'FC1', 'FCz', 'FC2', 'FC4', 'FC6', 'C5', 'C3', 'C1', 'Cz', 'C2', 'C4', 'C6', 'CP5', 'CP3', 'CP1', 'CPz', 'CP2', 'CP4', 'CP6', 'Fp1', 'Fpz', 'Fp2', 'AF7', 'AF3', 'AFz', 'AF4', 'AF8', 'F7', 'F5', 'F3', 'F1', 'Fz', 'F2', 'F4', 'F6', 'F8', 'FT7', 'FT8', 'T7', 'T8', 'T9', 'T10', 'TP7', 'TP8', 'P7', 'P5', 'P3', 'P1', 'Pz', 'P2', 'P4', 'P6', 'P8', 'PO7', 'PO3', 'POz', 'PO4', 'PO8', 'O1', 'Oz', 'O2', 'Iz']
     chs: 64 EEG
     custom_ref_applied: False
     dig: 67 items (3 Cardinal, 64 EEG)
     highpass: 8.0 Hz
     lowpass: 32.0 Hz
     meas_date: 2009-08-12 16:15:00 UTC
     nchan: 64
     projs: []
     sfreq: 160.0 Hz
    >
    '''