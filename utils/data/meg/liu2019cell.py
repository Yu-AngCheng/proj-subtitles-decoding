#!/usr/bin/env python3
"""
Created on 15:17, Dec. 27th, 2022

@author: Norbert Zheng
"""
import os
import numpy as np
# local dep
if __name__ == "__main__":
    import os, sys
    sys.path.insert(0, os.path.join(os.pardir, os.pardir, os.pardir))
from utils import DotDict

__all__ = [
    "load_liu2019cell",
]

# def load_liu2019cel func
def load_liu2019cell(path_data, subjects_allowed=None):
    """
    Load data from specified data path.
    :param path_data: path - The path of specified data.
    :param subjects_allowed: (n_subjects[list],) - The allowed subjects to load data.
    :return data: (n_data[list],) - The whole dataset.
    """
    # Initialize path_subjects.
    path_subjects = DotDict()
    for root, dirs, files in os.walk(path_data, topdown=False):
        for file_i in files:
            subject_name = os.path.splitext(file_i)[0].split("_")[0]
            session_idx = os.path.splitext(file_i)[0].split("_")[1]
            data_type = os.path.splitext(file_i)[0].split("_")[-1]
            if not hasattr(path_subjects, subject_name):
                path_subjects[subject_name] = DotDict()
            if not hasattr(path_subjects[subject_name], session_idx):
                path_subjects[subject_name][session_idx] = DotDict({"data":None,"montage":None,})
            if data_type == "data":
                path_subjects[subject_name][session_idx].data = os.path.join(root, file_i)
            else:
                path_subjects[subject_name][session_idx].montage = os.path.join(root, file_i)
    # Initialize data, then create the whole dataset.
    data = []
    subjects_allowed = subjects_allowed if subjects_allowed is not None else\
        np.arange(len(path_subjects.keys())).tolist()
    for subject_id, path_subject in enumerate(path_subjects.values()):
        if subject_id not in subjects_allowed: continue
        data_subject = _load_subject(path_subject)
        for data_idx in range(len(data_subject)):
            data_subject[data_idx]["subject_id"] = subject_id
        data.extend(data_subject)
    # Return the final `data`.
    return data

# def _load_subject func
def _load_subject(path_subject):
    """
    Load data from specified subject path.
    :param path_subject: DotDict - A `path_subject` DotDict, which contains [data,montage].
    """
    # Initialize data_subject.
    data_subject = []
    for path_session in path_subject.values():
        montage_session = np.load(path_session.montage, allow_pickle=True)[:,:2]
        data_session = np.load(path_session.data, allow_pickle=True)[()]
        for stimulus_type in data_session.keys():
            for meg_i in data_session[stimulus_type]["meg"]:
                data_i = DotDict({"data":meg_i[:,:-1].T,"label":stimulus_type.lower().replace(" ", ""),
                    "stimulus":data_session[stimulus_type]["stim"],
                    "channel_locations":montage_session,})
                data_subject.append(data_i)
    # Return the final `data_subject`.
    return data_subject

if __name__ == "__main__":
    # macro
    base = os.path.join(os.getcwd(), os.pardir, os.pardir, os.pardir)
    path_data = os.path.join(base, "data", "meg.liu2019cell")

    # Load data from specified data.
    data = load_liu2019cell(path_data, subjects_allowed=[0,])

