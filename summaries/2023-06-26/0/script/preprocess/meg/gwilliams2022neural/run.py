
import os, re
import numpy as np
# local dep
if __name__ == "__main__":
    import os, sys
    sys.path.insert(0, os.path.join(os.pardir, os.pardir, os.pardir))
    from session import preprocess_session
else:
    from preprocess.meg.gwilliams2022neural.session import preprocess_session
from utils import DotDict

__all__ = [
    "preprocess_run",
]

# def preprocess_run func
def preprocess_run(path_run, path_stimulus, events_=None):
    """
    The whole pipeline to preprocess meg data of specified run.
    :param path_run: path - The path of specified run.
    :param path_stimulus: path - The path of stimulus.
    :param events_: (n_sessions,) - The list of reference [train,test] events.
    :return dataset: tuple - The list of [train,test] data items.
    :return events: tuple - The corresponding [train,test] events.
    """
    # Initialize the path of sessions.
    path_run_meg = os.path.join(path_run, "meg")
    path_sessions = [os.path.join(path_run_meg, path_i) for path_i in os.listdir(path_run_meg)\
        if path_i.endswith(".con")]; path_sessions.sort()
    pattern = r"task-(\d+)"; pattern = re.compile(pattern)
    # Get the corresponding dataset & events of each session.
    dataset = []; events = DotDict()
    for session_idx in range(len(path_sessions)):
        # Initialize the task type of current session, check whether events_ has specified task type.
        task_type_i = "-".join(["task", pattern.findall(os.path.basename(path_sessions[session_idx]))[0]])
        if events_ is not None and not hasattr(events_, task_type_i): continue
        # Load meg data from specified run session.
        dataset_i, events_i = preprocess_session(path_sessions[session_idx], path_stimulus,
            events_=None if events_ is None else events_[task_type_i])
        dataset.append(dataset_i); events[task_type_i] = events_i
    n_events_train = np.sum([len(events_session_i[0]) for events_session_i in events.values()])
    n_events_test = np.sum([len(events_session_i[1]) for events_session_i in events.values()])
    n_train = np.sum([len(dataset_session_i[0]) for dataset_session_i in dataset])
    n_test = np.sum([len(dataset_session_i[1]) for dataset_session_i in dataset])
    assert n_events_train == n_train and n_events_test == n_test
    print((
        "INFO: Get {:d} samples ({:.2f}%) for trainset, and {:d} samples ({:.2f}%)"+
        " for testset in preprocess.meg.gwilliams2022neural.run."
    ).format(n_train, n_train/(n_train+n_test)*100., n_test, n_test/(n_train+n_test)*100.))
    # Return the final `dataset` & `events`.
    return dataset, events

if __name__ == "__main__":
    # macro
    base = os.path.join(os.getcwd(), os.pardir, os.pardir, os.pardir)
    path_dataset = os.path.join(base, "data", "meg.gwilliams2022neural")
    path_run1 = os.path.join(path_dataset, "sub-01", "ses-0")
    path_run2 = os.path.join(path_dataset, "sub-04", "ses-1")
    path_stimulus = os.path.join(path_dataset, "stimuli")

    # Initialize random seed.
    np.random.seed(42)

    # Preprocess the specified subject run.
    dataset1, events1 = preprocess_run(path_run1, path_stimulus)
    dataset2, events2 = preprocess_run(path_run2, path_stimulus, events_=events1)

