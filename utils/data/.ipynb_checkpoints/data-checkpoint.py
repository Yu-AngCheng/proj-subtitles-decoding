import pickle


__all__ = [
    "save_pickle",
    "load_pickle",
]


# def save_pickle func
def save_pickle(fname, obj):
    """
    Save object to pickle file.
    :param fname: The file name to save object.
    :param obj: The object to be saved.
    """
    with open(fname, "wb") as f:
        pickle.dump(obj, f)


# def load_pickle func
def load_pickle(fname):
    """
    Load object from pickle file.
    :param fname: The file name to load object.
    :return obj: The object loaded from file.
    """
    with open(fname, "rb") as f:
        obj = pickle.load(f)
    return obj

