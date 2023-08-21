import os
import numpy as np
import torch.nn.functional as F
import pickle

# local dep
if __name__ == "__main__":
    import os, sys
    sys.path.insert(0, os.path.join(os.pardir, os.pardir, os.pardir))
from utils import DotDict
from utils.data import load_pickle

__all__ = [
    "load_gwilliams2022neural_origin",
]

# def donwnsampling array
def downsample_array(array, original_sample_rate=22050, target_sample_rate=16000):
    original_length = len(array)
    target_length = int(original_length * target_sample_rate / original_sample_rate)
    indices = np.linspace(0, original_length - 1, target_length).astype(int)
    downsampled_array = np.interp(indices, np.arange(original_length), array)
    return downsampled_array

'''

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
    # dataset_train = []; dataset_test = []
    num_dataset_train = 0
    num_dataset_test = 0
    subjects_allowed = subjects_allowed if subjects_allowed is not None else np.arange(len(path_subjects)).tolist()

    target_audio_rate=16000

    for subject_id, path_subject in enumerate(path_subjects):
        if subject_id not in subjects_allowed: continue
        dataset_i = load_pickle(os.path.join(path_subject, "dataset.origin"))
        dataset_train_i, dataset_test_i = dataset_i.train, dataset_i.test

        for data_train_idx in range(len(dataset_train_i)):
            dataset_train_i[data_train_idx]["subject_id"] = subject_id
            if dataset_train_i[data_train_idx]["audio"][1].shape[0] >= 16000:
                dataset_train_i[data_train_idx]["audio"][0] = target_audio_rate
                dataset_train_i[data_train_idx]["audio"][1] = downsample_array(dataset_train_i[data_train_idx]["audio"][1])
                filename = os.path.join(path_data, "transition_train_data"+str(num_dataset_train)+".pickle")
                with open(filename, 'wb') as file:
                    pickle.dump(dataset_train_i[data_train_idx], file)

                num_dataset_train = num_dataset_train + 1

        for data_test_idx in range(len(dataset_test_i)):
            dataset_test_i[data_test_idx]["subject_id"] = subject_id
            if dataset_test_i[data_test_idx]["audio"][1].shape[0] >= 16000:
                dataset_test_i[data_test_idx]["audio"][0] = target_audio_rate
                dataset_test_i[data_test_idx]["audio"][1] = downsample_array(dataset_test_i[data_test_idx]["audio"][1])
                filename = os.path.join(path_data, "transition_test_data"+str(num_dataset_test)+".pickle")
                with open(filename, 'wb') as file:
                    pickle.dump(dataset_train_i[data_train_idx], file)

                num_dataset_test = num_dataset_test + 1

        # dataset_train.extend(dataset_train_i); dataset_test.extend(dataset_test_i)

    # removve data with audo length < 1s
    # dataset_train = [data_i for data_i in dataset_train if data_i["audio"][1].shape[0] >= 16000]
    # dataset_test = [data_i for data_i in dataset_test if data_i["audio"][1].shape[0] >= 16000]

    # Log information related to loaded dataset.
    print((
        "INFO: Get {:d} samples in trainset, and {:d} samples in testset in utils.data.meg.gwilliams2022neural."
    ).format(num_dataset_train, num_dataset_test))
    # print((
    #     "INFO: Get {:d} samples in trainset, and {:d} samples in testset in utils.data.meg.gwilliams2022neural."
    # ).format(len(dataset_train), len(dataset_test)))
    # Return the final `dataset_train` & `dataset_test`.
    # return dataset_train, dataset_test
    return 


if __name__ == "__main__":
    # macro
    base = os.path.join(os.getcwd(), os.pardir, os.pardir, os.pardir)
    path_data = os.path.join(base, "data", "meg.gwilliams2022neural")
    # Load data from specified data.
    n_subjects = 3
    subjects_allowed = list(range(n_subjects))
    load_gwilliams2022neural_origin(path_data, subjects_allowed=subjects_allowed)
'''

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
    
    # removve data with audo length < 1s
    dataset_train = [data_i for data_i in dataset_train if data_i["audio"][1].shape[0] >= 16000]
    dataset_test = [data_i for data_i in dataset_test if data_i["audio"][1].shape[0] >= 16000]

    # Log information related to loaded dataset.
    print((
        "INFO: Get {:d} samples in trainset, and {:d} samples in testset in utils.data.meg.gwilliams2022neural."
    ).format(len(dataset_train), len(dataset_test)))
    # Return the final `dataset_train` & `dataset_test`.
    return dataset_train, dataset_test

# def donwnsampling array
def downsample_array(array, original_sample_rate=22050, target_sample_rate=16000):
    original_length = len(array)
    target_length = int(original_length * target_sample_rate / original_sample_rate)
    indices = np.linspace(0, original_length - 1, target_length).astype(int)
    downsampled_array = np.interp(indices, np.arange(original_length), array)
    return downsampled_array

if __name__ == "__main__":
    # macro
    base = os.path.join(os.getcwd(), os.pardir, os.pardir, os.pardir)
    path_data = os.path.join(base, "data", "meg.gwilliams2022neural")
    # Load data from specified data.
    n_subjects = 2
    subjects_allowed = list(range(n_subjects))
    dataset_train, dataset_test = load_gwilliams2022neural_origin(path_data, subjects_allowed=subjects_allowed)
    # print(data[0][0].keys()) # dict_keys(['name', 'audio', 'data', 'chan_pos', 'subject_id'])

    # downsampling audio rate for each data
    target_audio_rate=16000
    for data_i in dataset_train:
        data_i["audio"][0] = target_audio_rate
        data_i["audio"][1] = downsample_array(data_i["audio"][1])
    for data_i in dataset_test:
        data_i["audio"][0] = target_audio_rate
        data_i["audio"][1] = downsample_array(data_i["audio"][1])

    # save data with pickle
    filename = os.path.join(path_data, "meg/transition_train_data.pkl")
    with open(filename, 'wb') as file:
        pickle.dump(dataset_train, file)
    
    filename = os.path.join(path_data, "meg/transition_test_data.pkl")
    with open(filename, 'wb') as file:
        pickle.dump(dataset_test, file)