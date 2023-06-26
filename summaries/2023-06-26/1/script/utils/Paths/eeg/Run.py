#!/usr/bin/env python3
"""
Created on 21:11, Dec. 23rd, 2022

@author: Norbert Zheng
"""
import os
import numpy as np
import pandas as pd
from bidict import bidict
# local dep
if __name__ == "__main__":
    import os, sys
    sys.path.insert(0, os.path.join(os.pardir, os.pardir, os.pardir))
from utils import DotDict

__all__ = [
    "Run",
]

class Run:
    """
    `Run` class of the specified run.
    """

    def __init__(self, base):
        """
        Initialize `Run` object of specified run. The run path contains the following files:
         - behavior: The behavior files of the specified run.
         - eeg: The eeg files of the specified run.
         - standard-1020-cap64.locs: The configuration of locations corresponding to each eeg channel.
        :param base: The base path of specified run.
        """
        # Initialize parameters.
        self.base = base
        # Initialize variables.
        self._init_metadatas()

    """
    init funcs
    """
    # def _init_metadatas func
    def _init_metadatas(self):
        """
        Initialize the metadata of specified run. All metadatas are loaded from behavior files.
        Note: Only works for [004,...] data, due to the column names of behavior csv table.
        """
        # Initialize the path of montage.
        self.montage = os.path.join(self.base, [path_i for path_i in os.listdir(self.base) if path_i.endswith(".locs")][0])
        # Initialize the path of behavior, and load all behavior files.
        path_behavior = os.path.join(self.base, "behavior")
        behaviors = [pd.read_csv(os.path.join(path_behavior, path_i))\
            for path_i in os.listdir(path_behavior) if path_i.endswith(".csv")]
        # Initialize the markers dict.
        image_markers = [behavior_i[["visual_stimuli", "visual_marker"]].dropna(axis=0, how="any").values.tolist()\
            for behavior_i in behaviors if "visual_stimuli" in behavior_i.columns and "visual_marker" in behavior_i.columns]
        image_markers = bidict([marker_pair_i for image_markers_session in image_markers\
            for marker_pair_i in image_markers_session])
        audio_markers = [behavior_i[["audio_stimuli", "audio_marker"]].dropna(axis=0, how="any").values.tolist()\
            for behavior_i in behaviors if "audio_stimuli" in behavior_i.columns and "audio_marker" in behavior_i.columns]
        audio_markers = bidict([marker_pair_i for audio_markers_session in audio_markers\
            for marker_pair_i in audio_markers_session])
        resp_markers = [behavior_i[["resp_stimuli", "resp_marker"]].dropna(axis=0, how="any").values.tolist()\
            for behavior_i in behaviors if "resp_stimuli" in behavior_i.columns and "resp_marker" in behavior_i.columns]
        resp_markers = bidict([marker_pair_i for resp_markers_session in resp_markers\
            for marker_pair_i in resp_markers_session]) if len(resp_markers) > 0 else None
        '''
        tmr_audio_markers =  [behavior_i[["Audio"]].dropna(axis=0, how="any").values.tolist()\
            for behavior_i in behaviors if "Audio" in behavior_i.columns]
        tmr_audio_markers = bidict([marker_pair_i for tmr_audio_markers_session in tmr_audio_markers\
            for marker_pair_i in tmr_audio_markers_session]) if len(tmr_audio_markers) > 0 else None
        '''
        self.markers = DotDict({"image":image_markers,"audio":audio_markers," resp":resp_markers})

        # Initialize the eeg dict.
        # Note: We should ensure the time sequence of different type behavior files.

        image_audio_tpoints = [behavior_i[["i_test.started", "a_test.started", "k_resp.started"]].dropna(axis=0, how="any").values\
            for behavior_i in behaviors if "i_test.started" in behavior_i.columns and\
            "a_test.started" in behavior_i.columns and "k_resp.started" in behavior_i.columns]
        # print(np.array(image_audio_tpoints).shape) （4 600 3）
        if len(image_audio_tpoints) > 0:
            # Get the mean difference of each duration.
            image_audio_tpoints = [np.round(np.mean(
                image_audio_tpoints_session[:,-2:] - image_audio_tpoints_session[:,:2]
            , axis=0), decimals=1) for image_audio_tpoints_session in image_audio_tpoints]
            # Only keep specified ones, to avoid negative values.
            image_audio_tpoints = [image_audio_tpoints_session for image_audio_tpoints_session in image_audio_tpoints\
                if np.min(image_audio_tpoints_session) > 0]
            image_audio_tpoints = np.stack(image_audio_tpoints, axis=0)
            # Check whether all differences are the same.
            assert np.sum(image_audio_tpoints - image_audio_tpoints[0,:]) == 0, ((
                "ERROR: image_audio_tpoints {} are not all the same along axis-0."
            ).format(image_audio_tpoints))
            image_audio_tpoints = np.mean(image_audio_tpoints, axis=0)
        else:
            image_audio_tpoints = None
        image_audio_eeg = DotDict({
            "image":{"tmin":-0.2,"tmax":image_audio_tpoints[0],},
            "audio":{"tmin":-0.2,"tmax":image_audio_tpoints[1],},
        }) if image_audio_tpoints is not None else None

        audio_image_tpoints = [behavior_i[["a_test.started", "i_test.started", "k_resp.started"]].dropna(axis=0, how="any").values\
            for behavior_i in behaviors if "a_test.started" in behavior_i.columns and\
            "i_test.started" in behavior_i.columns and "k_resp.started" in behavior_i.columns]

        if len(audio_image_tpoints) > 0:
            # Get the mean difference of each duration.
            audio_image_tpoints = [np.round(np.mean(
                audio_image_tpoints_session[:,-2:] - audio_image_tpoints_session[:,:2]
            , axis=0), decimals=1) for audio_image_tpoints_session in audio_image_tpoints]

            # Only keep specified ones, to avoid negative values.
            audio_image_tpoints = [audio_image_tpoints_session for audio_image_tpoints_session in audio_image_tpoints\
                if np.min(audio_image_tpoints_session) > 0]
            audio_image_tpoints = np.stack(audio_image_tpoints, axis=0)
            # Check whether all differences are the same.
            assert np.sum(audio_image_tpoints - audio_image_tpoints[0,:]) == 0, ((
                "ERROR: audio_image_tpoints {} are not all the same along axis-0."
            ).format(audio_image_tpoints))
            audio_image_tpoints = np.mean(audio_image_tpoints, axis=0)   
        else:
            audio_image_tpoints = None

        audio_image_eeg = DotDict({
            "audio":{"tmin":-0.2,"tmax":audio_image_tpoints[0],},
            "image":{"tmin":-0.2,"tmax":audio_image_tpoints[1],},
        }) if audio_image_tpoints is not None else None
        
        # we think that the duration of resp triggered by audio in sleep also is the same as that in awake, so we directly use audio_image_tpoints[0]
        tmr_eeg = DotDict({
            "audio":{"tmin":-0.2,"tmax":audio_image_tpoints[0],},
        })

        # Note: If the value is not None, the corresponding key is a valid session type.
        self.eeg = DotDict({"image-audio":image_audio_eeg,"audio-image":audio_image_eeg,"tmr":tmr_eeg})

if __name__ == "__main__":
    # macro
    base = os.path.join(os.getcwd(), os.pardir, os.pardir, os.pardir)
    # path_data = os.path.join(base, "data")
    # path_subject = os.path.join(path_data, "005")
    # path_run = os.path.join(path_subject, "20221223")
    path_run = os.path.join(base, "data", "eeg", "005", "20221223")
    # Instantiate Run.
    run_inst = Run(path_run)

