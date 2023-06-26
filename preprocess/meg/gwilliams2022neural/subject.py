import os
import numpy as np
# local dep
if __name__ == "__main__":
    import os, sys
    sys.path.insert(0, os.path.join(os.pardir, os.pardir, os.pardir))
    from run import preprocess_run
else:
    from preprocess.meg.gwilliams2022neural.run import preprocess_run

__all__ = [
    "preprocess_subject",
]

# def preprocess_subject func
def preprocess_subject(path_subject, path_stimulus, events_=None):
    """
    The whole pipeline to preprocess meg data of specified subject.
    :param path_subject: path - The path of specified subject.
    :param path_stimulus: path - The path of stimulus.
    :param events_: (n_sessions,) - The list of reference [train,test] events.
    :return dataset: tuple - The list of [train,test] data items of one run.
    :return events: tuple - The corresponding [train,test] events of one session.
    """
    # Initialize the path of runs.
    path_runs = [os.path.join(path_subject, path_i) for path_i in os.listdir(path_subject)\
        if os.path.isdir(os.path.join(path_subject, path_i))]; path_runs.sort()
    # Get the corresponding dataset & events of each run.
    dataset = []
    for run_idx in range(len(path_runs)):
        dataset_i, events_i = preprocess_run(path_runs[run_idx], path_stimulus, events_=events_)
        dataset.append(dataset_i); events_ = events_ if events_ is not None else events_i
    n_train = [np.sum([len(dataset_session[0]) for dataset_session in dataset_run]) for dataset_run in dataset]
    n_test = [np.sum([len(dataset_session[1]) for dataset_session in dataset_run]) for dataset_run in dataset]
    print((
        "INFO: Get {:d} samples ({:.2f}%) for trainset with {} segments ({:.2f}% on average), and {:d} samples"+
        " ({:.2f}%) for testset with {} segments ({:.2f}% on average) in preprocess.meg.gwilliams2022neural.subject."
    ).format(np.sum(n_train), np.sum(n_train)/(np.sum(n_train)+np.sum(n_test))*100.,
             n_train, int(np.mean(n_train))/(int(np.mean(n_train))+int(np.mean(n_test)))*100.,
             np.sum(n_test), np.sum(n_test)/(np.sum(n_train)+np.sum(n_test))*100.,
             n_test, int(np.mean(n_test))/(int(np.mean(n_train))+int(np.mean(n_test)))*100.,
    ))
    # Return the final `dataset` & `events`.
    return dataset, events_

if __name__ == "__main__":
    # macro
    base = os.path.join(os.getcwd(), os.pardir, os.pardir, os.pardir)
    path_dataset = os.path.join(base, "data", "meg.gwilliams2022neural")
    path_subject1 = os.path.join(path_dataset, "sub-04")
    path_subject2 = os.path.join(path_dataset, "sub-01")
    path_stimulus = os.path.join(path_dataset, "stimuli")

    # Initialize random seed.
    np.random.seed(42)

    # Preprocess the specified subject.
    dataset1, events1 = preprocess_subject(path_subject1, path_stimulus)
    dataset2, events2 = preprocess_subject(path_subject2, path_stimulus, events_=events1)

