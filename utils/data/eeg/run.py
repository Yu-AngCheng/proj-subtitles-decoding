#!/usr/bin/env python3
"""
Created on 19:59, Dec. 25th, 2022

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
from utils.Paths.eeg import Run

__all__ = [
    "load_run",
]

# def load_run func
def load_run(path_run, session_type="tmr"):
    """
    Load data from specified run.
    :param path_run: path - The path of specified run.
    :param session_type: str - The type of session.
    image-audio
    """

    # Load data from specified run.
    data = load_pickle(os.path.join(path_run, "dataset"))[session_type]; np.random.shuffle(data)
    if session_type ==["tmr"]:
        audio_labels = [data_i.audio.name.split("-")[1]\
            if data_i.audio is not None else None for data_i in data]
        labels = list(set(audio_labels)); labels.sort()
        y_audio = np.array([labels.index(audio_label_i) for audio_label_i in audio_labels], dtype=np.int64)
        X_audio = np.array([data_i.audio.data.T for data_i in data], dtype=np.float32)
        return DotDict({"image":None,"audio":X_audio,}), DotDict({"image":None,"audio":y_audio,})
    elif session_type in ["image-audio", "audio-image"]:
        # Get the corresponding label set.
        image_labels = [data_i.image.name.split("-")[1]\
            if data_i.image is not None else None for data_i in data]
        audio_labels = [data_i.audio.name.split("-")[1]\
            if data_i.audio is not None else None for data_i in data]
        # Check whether have both [image,audio] data.
        # TODO: Add paired [image,audio] raw data.
        
        if None in set(image_labels):
            assert ("image" == session_type.split("-")[1]) and (not None in set(audio_labels))
            labels = list(set(audio_labels)); labels.sort()
            y_audio = np.array([labels.index(audio_label_i) for audio_label_i in audio_labels], dtype=np.int64)
            X_audio = np.array([data_i.audio.data.T for data_i in data], dtype=np.float32)
            return DotDict({"image":None,"audio":X_audio,}), DotDict({"image":None,"audio":y_audio,})
        elif None in set(audio_labels):
            assert ("audio" == session_type.split("-")[1]) and (not None in set(image_labels))
            labels = list(set(image_labels)); labels.sort()
            y_image = np.array([labels.index(image_label_i) for image_label_i in image_labels], dtype=np.int64)
            X_image = np.array([data_i.image.data.T for data_i in data], dtype=np.float32)
            return DotDict({"image":X_image,"audio":None,}), DotDict({"image":y_image,"audio":None,})
        else:
            assert set(image_labels) == set(audio_labels)
            labels = list(set(image_labels)); labels.sort()
            y_image = np.array([labels.index(image_label_i) for image_label_i in image_labels], dtype=np.int64)
            X_image = np.array([data_i.image.data.T for data_i in data], dtype=np.float32)
            y_audio = np.array([labels.index(audio_label_i) for audio_label_i in audio_labels], dtype=np.int64)
            X_audio = np.array([data_i.audio.data.T for data_i in data], dtype=np.float32)
            return DotDict({"image":X_image,"audio":X_audio,}), DotDict({"image":y_image,"audio":y_audio,})


if __name__ == "__main__":
    # macro
    base = os.path.join(os.getcwd(), os.pardir, os.pardir, os.pardir)
    path_run = os.path.join(base, "data", "eeg", "005", "20221223")

    # Load data from specified run.
    data = load_run(path_run)

