import os
import numpy as np
from preprocess.meg.gwilliams2022neural.subject import preprocess_subject
from utils import DotDict
from utils.data import save_pickle

__all__ = [
    "create_dataset",
]


# def create_dataset func
def create_dataset(path_dataset):
    """
    Create the original dataset of `gwilliams2022neural` dataset.
    :param path_dataset: path - The path of `gwilliams2022neural` dataset.
    """
    # Initialize the path of subjects.
    path_subjects = [os.path.join(path_dataset, path_i) for path_i in os.listdir(path_dataset)\
        if os.path.isdir(os.path.join(path_dataset, path_i)) and path_i.startswith("sub")]; path_subjects.sort()
    path_stimulus = os.path.join(path_dataset, "stimuli")
    # Get the corresponding dataset & events of each subject.
    events_ = None; n_train = 0; n_test = 0
    for subject_idx in range(len(path_subjects)):
        # Preprocess the specified subject.
        dataset_subject_i, events_ = preprocess_subject(path_subjects[subject_idx], path_stimulus, events_=events_)
        # Flatten the original dataset to get [train,test]-set.
        trainset_subject_i = [data_train_i for dataset_run_i in dataset_subject_i\
            for dataset_session_i in dataset_run_i for data_train_i in dataset_session_i[0]]
        testset_subject_i = [data_test_i for dataset_run_i in dataset_subject_i\
            for dataset_session_i in dataset_run_i for data_test_i in dataset_session_i[1]]
        n_train += len(trainset_subject_i); n_test += len(testset_subject_i)
        # Save [train,test]-set to the corresponding dataset.
        save_pickle(os.path.join(path_subjects[subject_idx], "dataset.origin"),
            DotDict({"train":trainset_subject_i,"test":testset_subject_i,}))
        
        # Log information related to current subject.
        print((
            "INFO: Finish creating the original dataset of gwilliams2022neural dataset for subject {:d},"+
            " with {:d} samples ({:.2f}%) for trainset and {:d} samples ({:.2f}%) for testset saved to {}"+
            " in preprocess.meg.gwilliams2022neural.dataset."
        ).format(subject_idx,
            len(trainset_subject_i), len(trainset_subject_i)/(len(trainset_subject_i)+len(testset_subject_i))*100.,
            len(testset_subject_i), len(testset_subject_i)/(len(trainset_subject_i)+len(testset_subject_i))*100.,
            os.path.join(path_subjects[subject_idx], "dataset.origin")))
    print((
        "INFO: Finish creating the original dataset of gwilliams2022neural dataset, with {:d} samples ({:.2f}%)"+
        " for trainset and {:d} samples ({:.2f}%) for testset in preprocess.meg.gwilliams2022neural.dataset."
    ).format(n_train, n_train/(n_train+n_test)*100., n_test, n_test/(n_train+n_test)*100.))


if __name__ == "__main__":
    # macro
    base = os.path.join(os.getcwd())
    path_dataset = os.path.join(base, "data", "meg.gwilliams2022neural")

    # Initialize random seed.
    np.random.seed(42)

    # Create dataset of `gwilliams2022neural` dataset.
    create_dataset(path_dataset)

