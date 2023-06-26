#!/usr/bin/env python3
"""
Created on 20:46, Jan. 7th, 2023

@author: Norbert Zheng
"""
import os
import numpy as np
# local dep
if __name__ == "__main__":
    import os, sys
    sys.path.insert(0, os.path.join(os.pardir, os.pardir, os.pardir))
from utils import DotDict
from utils.data import load_pickle

__all__ = [
    "load_gwilliams2022neural_origin",
]

# def load_gwilliams2022neural_origin func
def load_gwilliams2022neural_origin(path_data, subjects_allowed=None):
    """
    Load the origin data from specified data path.
    :param path_data: path - The path of specified data.
    :param subjects_allowed: (n_subjects[list],) - The allowed subjects to load data.
    :return dataset_train: (n_train[list],) - The train dataset.
    :return dataset_test: (n_test[list],) - The test dataset.
    """
    # Initialize path_subjects.
    path_subjects = [os.path.join(path_data, path_i) for path_i in os.listdir(path_data)\
        if os.path.isdir(os.path.join(path_data, path_i)) and path_i.startswith("sub-")]; path_subjects.sort()
    # Initialize dataset_train & dataset_test.
    dataset_train = []; dataset_test = []
    subjects_allowed = subjects_allowed if subjects_allowed is not None else np.arange(len(path_subjects)).tolist()
    for subject_id, path_subject in enumerate(path_subjects):
        if subject_id not in subjects_allowed: continue
        dataset_i = load_pickle(os.path.join(path_subject, "dataset.origin"))
        dataset_train_i, dataset_test_i = dataset_i.train, dataset_i.test
        for data_train_idx in range(len(dataset_train_i)):
            dataset_train_i[data_train_idx]["subject_id"] = subject_id
        for data_test_idx in range(len(dataset_test_i)):
            dataset_test_i[data_test_idx]["subject_id"] = subject_id
        dataset_train.extend(dataset_train_i); dataset_test.extend(dataset_test_i)
    # Log information related to loaded dataset.
    print((
        "INFO: Get {:d} samples in trainset, and {:d} samples in testset in utils.data.meg.gwilliams2022neural."
    ).format(len(dataset_train), len(dataset_test)))
    # Return the final `dataset_train` & `dataset_test`.
    return dataset_train, dataset_test

if __name__ == "__main__":
    # macro
    base = os.path.join(os.getcwd(), os.pardir, os.pardir, os.pardir)
    path_data = os.path.join(base, "data", "meg.gwilliams2022neural")

    # Load data from specified data.
    data = load_gwilliams2022neural_origin(path_data, subjects_allowed=[0,1,])

