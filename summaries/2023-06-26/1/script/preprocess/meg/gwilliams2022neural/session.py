
import os, mne
import itertools
import numpy as np
import pandas as pd
from bidict import bidict
from scipy.io.wavfile import read
from nltk.tokenize import sent_tokenize
from sklearn.preprocessing import robust_scale

# local dep
if __name__ == "__main__":
    import os, sys
    sys.path.insert(0, os.path.join(os.pardir, os.pardir, os.pardir))
from utils import DotDict

__all__ = [
    "preprocess_session",
]

# def load_data func
def load_data(path_meg, resample_rate=120):
    """
    Load meg data from specified session.
    :param path_meg: path - The path of specified session meg data.
    :param resample_rate: int - The rate of resample.
    :return data: object - The modified `mne.io.brainvision.brainvision.RawBrainVision` object.
    :return positions: (n_channels, 2) - The positions of each meg sensor.
    :return events: `pd.DataFrame` - The data frame of events.
    """
    ## Prepare for loading data.
    # Initialize the path of run and the types of [run,session].
    path_run = os.path.dirname(path_meg)
    run_type = "_".join(os.path.splitext(os.path.basename(path_meg))[0].split("_")[:2])
    session_type = "_".join(os.path.splitext(os.path.basename(path_meg))[0].split("_")[:3])
    # Initialize paths of other configurations.
    path_elp = os.path.join(path_run, "_".join([run_type, "acq-ELP_headshape.pos"]))
    path_hsp = os.path.join(path_run, "_".join([run_type, "acq-HSP_headshape.pos"]))
    path_channels = os.path.join(path_run, "_".join([session_type, "channels.tsv"]))
    path_events = os.path.join(path_run, "_".join([session_type, "events.tsv"]))
    path_markers = os.path.join(path_run, "_".join([session_type, "markers.mrk"]))
    # Load elp & hsp positions from specified pos files.
    # pos_elp - (8, 3)
    pos_elp = pd.read_csv(path_elp, sep="\s+", comment="%", header=None).values
    # pos_hsp - (n_points, 3)
    pos_hsp = pd.read_csv(path_hsp, sep="\s+", comment="%", header=None).values

    ## Load meg data.
    # Load meg data from specified session.
    # Note: We cannot load all data with [mrk,elp,hsp]. Otherwise, it will raise memory error:
    # >>> numpy.core._exceptions._ArrayMemoryError: Unable to allocate 524. TiB for an array
    # >>> with shape (71987334686253,) and data type int64
    data = mne.io.read_raw_kit(path_meg, preload=True, verbose="WARNING")
    # Resample data down to 120Hz, with the following warning:
    # >>> Trigger channel has a non-zero initial value of 255 (consider using initial_event=True to detect this event)
    # >>> 14 events found
    # >>> Event IDs: [255]
    # >>> Trigger channel has a non-zero initial value of 255 (consider using initial_event=True to detect this event)
    # >>> 5 events found
    # >>> Event IDs: [255]
    # >>> RuntimeWarning: Resampling of the stim channels caused event information to become unreliable.
    # >>> Consider finding events on the original data and passing the event matrix as a parameter.
    data.resample(sfreq=resample_rate, verbose="WARNING")

    ## Initialize channels, only keep mag channels.
    # Load channels from specified channels file.
    channels = pd.read_csv(path_channels, sep="\t")
    # Initialize channel types.
    # Note: As `data` already contains all information related to channel types,
    # we donot have to set the channel types of `data` using `channels`.
    # Initialize montage of `data`.
    # Note: As `data` already contains information related to montage,
    # we donot have to set the montage of `data`.
    # Remove bad channels, only keep mag channels.
    # >>> Parameters of `mne.pick_types`.
    # >>> meg: bool | str
    # >>>   If True include MEG channels. If string it can be 'mag', 'grad',
    # >>>   'planar1' or 'planar2' to select only magnetometers, all
    # >>>   gradiometers, or a specific type of gradiometer.
    # >>> stim : bool
    # >>>   If True include stimulus channels.
    # >>> ref_meg : bool | str
    # >>>   If True include CTF / 4D reference channels. If 'auto', reference
    # >>>   channels are included if compensations are present and ``meg`` is
    # >>>   not False. Can also be the string options for the ``meg``
    # >>>   parameter.
    # >>> misc : bool
    # >>>   If True include miscellaneous analog channels.
    data.pick(picks=mne.pick_types(data.info, meg=True, exclude=["bads"]))
    # Get layout from `data`, only keep [x,y]-positions.
    # (208, 2)
    positions = mne.channels.find_layout(data.info).pos[:,:2]

    ## Load events from specified session.
    # Load events from specified session events file.
    events = pd.read_csv(path_events, sep="\t")

    # Return the final `data` & `positions` & `events`.
    return data, positions, events

# def split_events func
def split_events(events, path_stimulus, split_duration=None, train_shift=None):
    """
    Split events according to fixed [train,validation,test]-ratio.
    :param events: `pd.DataFrame` - The loaded events.
    :param path_stimulus: path - The path of stimulus.
    :param split_duration: float - The minimum duration of each split.
    :param train_shift: float - The time shift between near train segments.
    :return events_train: (n_train[list],) - The list of train set, a 3s-segment starting from onset.
    :return events_test: (n_test[list],) - The list of test set, a 3s-segment starting from 500ms before onset.
    """
    ## Prepare for splitting data.
    # Convert events from `pd.DataFrame` to `DotDict`.
    # events - (n_events[list],)
    events_ = []
    for event_idx, event_i in events.iterrows():
        events_.append(DotDict({
            "onset": event_i["onset"],
            "duration": event_i["duration"],
            "trial_type": eval(event_i["trial_type"]),
            "value": event_i["value"],
            "sample": event_i["sample"],
        }))
    events = events_
    # Construct words from events, separated by audio slices.
    events_words = []; events_words_i = []
    durations_sound = []; start_sound = 0.; stop_sound = 0.
    for event_i in events:
        if event_i.trial_type.kind == "sound":
            events_words.append(events_words_i); events_words_i = []
            durations_sound.append([start_sound, stop_sound]); start_sound = np.round(event_i.onset, decimals=3)
        elif event_i.trial_type.kind == "word":
            events_words_i.append(DotDict({
                "onset": event_i.onset,
                "duration": event_i.duration,
                "trial_type": DotDict({
                    "story": event_i.trial_type.story,
                    "story_piece": int(event_i.trial_type.sound_id),
                    "start": event_i.trial_type.start,
                    "word": event_i.trial_type.word,
                    "condition": event_i.trial_type.condition,
                    "word_index": int(event_i.trial_type.word_index),
                }),
                "value": event_i.value,
                "sample": event_i.sample,
            })); stop_sound = np.round(event_i.onset + event_i.duration, decimals=3)
    events_words.append(events_words_i); durations_sound.append([start_sound, stop_sound])
    events_words = [events_words_i for events_words_i in events_words if len(events_words_i) > 0]
    durations_sound = [duration_sound_i for duration_sound_i in durations_sound if duration_sound_i[1]-duration_sound_i[0] > 0.]
    # Use sentence to separate words in each audio slice.
    events_sentences = []
    for events_words_i in events_words:
        events_sentences_i = []; event_sentence_i = []
        for event_word_i in events_words_i:
            if len(event_sentence_i) > 0 and event_word_i.trial_type.word_index <= event_sentence_i[-1].trial_type.word_index:
                events_sentences_i.append(event_sentence_i)
                event_sentence_i = []
            event_sentence_i.append(event_word_i)
        events_sentences_i.append(event_sentence_i)
        events_sentences_i = [event_sentence_i for event_sentence_i in events_sentences_i if len(event_sentence_i) > 0]
        events_sentences.append(events_sentences_i)
    # Check whether the number of sentences are consistent.
    story_type = events_sentences[0][0][0].trial_type.story
    path_stimulus_text = os.path.join(path_stimulus, "text", story_type.lower()+".txt")
    story_content = open(path_stimulus_text).read()
    # Note: There is no need to check whether the number of sentences are consistent. Maybe the
    # events have different rules to separate sentences. The number of sentences may be different.
    # More importantly, whether the number of sentences are consistent donot affect the dataset.
    # assert len(sent_tokenize(story_content)) == np.sum([len(events_sentences_i) for events_sentences_i in events_sentences])
    # Use `split_duration` to concat sentences.
    events_splits = []
    if split_duration is not None:
        for events_sentences_i in events_sentences:
            events_splits_i = []; start_idx = end_idx = 0
            while end_idx < len(events_sentences_i):
                time_stop = events_sentences_i[end_idx][-1].trial_type.start + events_sentences_i[end_idx][-1].duration
                time_start = events_sentences_i[start_idx][0].trial_type.start
                if time_stop - time_start > split_duration or end_idx == len(events_sentences_i) - 1:
                    event_split_i = []
                    for sequence_idx in range(start_idx, end_idx + 1):
                        event_split_i.extend(events_sentences_i[sequence_idx])
                    events_splits_i.append(event_split_i)
                    start_idx = end_idx + 1
                end_idx += 1
            events_splits.append(events_splits_i)
    else:
        events_splits = events_sentences
    starts_split = np.array([event_word_i.trial_type.start for events_splits_i in events_splits\
        for event_split_i in events_splits_i for event_word_i in event_split_i], dtype=np.float32)
    starts_sentence = np.array([event_word_i.trial_type.start for events_sentences_i in events_sentences\
        for event_sentence_i in events_sentences_i for event_word_i in event_sentence_i], dtype=np.float32)
    assert (starts_split == starts_sentence).all()

    ## Split (validation,test)-set from events_splits.
    # Sample splits to construct [validation,test]-set.
    events_splits_train = []; events_splits_test = []
    train_ratio = 0.7; validation_ratio = 0.2; test_ratio = 0.1
    assert train_ratio + validation_ratio + test_ratio > 0.
    ratio_sum = train_ratio + validation_ratio + test_ratio
    train_ratio /= ratio_sum; validation_ratio /= ratio_sum; test_ratio /= ratio_sum
    for events_splits_i in events_splits:
        events_splits_test_i = []
        for event_split_i in events_splits_i:
            if np.random.random() < (validation_ratio + test_ratio):
                events_splits_test_i.append(event_split_i)
        events_splits_test.append(events_splits_test_i)
    assert (np.array([len(events_splits_i) for events_splits_i in events_splits]) > 0).all()
    # Use word 500ms-onset to split 3s-segment.
    events_test = []
    for events_splits_test_i in events_splits_test:
        events_test_i = []
        for event_split_test_i in events_splits_test_i:
            for event_test_i in event_split_test_i:
                events_test_i.append(DotDict({
                    "story": event_test_i.trial_type.story,
                    "story_piece": event_test_i.trial_type.story_piece,
                    "onset": np.round(event_test_i.onset, decimals=3),
                    "start": np.round(event_test_i.trial_type.start, decimals=3),
                    "description": "-".join(["test", event_test_i.trial_type.word]),
                })); assert story_type == event_test_i.trial_type.story
        events_test.append(events_test_i)

    ## Split train-set according to events_test.
    # Use test-set to infer the duration of train-set.
    freq_ = 1000; train_mask = np.zeros((int(np.round(durations_sound[-1][-1]*freq_, decimals=0)),), dtype=np.bool_)
    for duration_sound_i in durations_sound:
        for sample_idx in range(int(np.round(duration_sound_i[0]*freq_, decimals=0)),
                                int(np.round(duration_sound_i[1]*freq_, decimals=0))):
            train_mask[sample_idx] = True
    n_samples = np.sum(train_mask)
    for events_test_i in events_test:
        for event_test_i in events_test_i:
            test_mask_i = np.zeros_like(train_mask, dtype=np.bool_)
            for sample_idx in range(int(np.round((event_test_i.onset-0.5)*freq_, decimals=0)),
                                    int(np.round((event_test_i.onset+2.5)*freq_, decimals=0))):
                if sample_idx < test_mask_i.shape[0]: test_mask_i[sample_idx] = True
            train_mask = train_mask & ~test_mask_i
    n_samples_train_possible = np.sum(train_mask); n_samples_test = n_samples - n_samples_train_possible
    # Get durations_train to check whether duration is long enough (greater than 3s).
    durations_train = []
    for group_type, group_i in itertools.groupby(enumerate(train_mask), lambda x: x[1]):
        if group_type:
            indices = list(zip(*group_i))[0]; durations_train.append([indices[0], indices[-1] + 1])
    durations_train = [[
        np.round(duration_train_i[0]/freq_, decimals=3),
        np.round(duration_train_i[1]/freq_, decimals=3)
    ] for duration_train_i in durations_train]
    train_mask_reconstr = np.zeros_like(train_mask, dtype=np.bool_)
    for duration_train_i in durations_train:
        for sample_idx in range(int(np.round(duration_train_i[0]*freq_, decimals=0)),
                                int(np.round(duration_train_i[1]*freq_, decimals=0))):
            train_mask_reconstr[sample_idx] = True
    assert (train_mask == train_mask_reconstr).all()
    durations_train = [duration_train_i for duration_train_i in durations_train\
        if duration_train_i[1] - duration_train_i[0] > 3.]
    n_samples_train_available = int(np.sum([duration_train_i[1]-duration_train_i[0]\
        for duration_train_i in durations_train]) * freq_)
    print((
        "INFO: Get {:.2f}% (selected from {:.2f}%) samples available for trainset, and {:.2f}%"+
        " samples available for testset in preprocess.meg.gwilliams2022neural.session."
    ).format((n_samples_train_available/n_samples)*100.,
        (n_samples_train_possible/n_samples)*100., (n_samples_test/n_samples)*100.))
    # Construct trainset from durations_train.
    # Note: train_shift have a scale of 1e-3.
    events_train = []
    train_shift = np.round(train_shift, decimals=3) if train_shift is not None else\
        np.round(np.mean([event_split_i[event_idx+1].onset - event_split_i[event_idx].onset\
            for events_splits_test_i in events_splits_test for event_split_i in events_splits_test_i\
            if len(event_split_i) > 1 for event_idx in range(len(event_split_i) - 1)]), decimals=3)
    # Note: We assume that the start of sound is exactly with `start=0`.
    for duration_train_i in durations_train:
        onset = duration_train_i[0]
        while np.round(onset + 3., decimals=3) <= duration_train_i[1]:
            sound_mask_i = np.array([duration_sound_i[0] <= onset and np.round(onset + 3., decimals=3) <= duration_sound_i[1]\
                for duration_sound_i in durations_sound], dtype=np.bool_)
            assert np.sum(sound_mask_i) == 1
            onset_sound_i = durations_sound[np.argmax(sound_mask_i)][0]
            events_train.append(DotDict({
                "story": story_type,
                "story_piece": np.argmax(sound_mask_i),
                "onset": np.round(onset, decimals=3),
                "start": np.round(onset - onset_sound_i, decimals=3),
                "description": "-".join(["train", "segment"]),
            }))
            onset = np.round(onset + train_shift, decimals=3)
    events_test = [event_i for events_test_i in events_test for event_i in events_test_i]
    n_events_train = len(events_train); n_events_test = len(events_test)
    print((
        "INFO: Get {:d} samples ({:.2f}%) for trainset with train_shift ({:.3f}s), and {:d}"+
        " samples ({:.2f}%) for testset in preprocess.meg.gwilliams2022neural.session."
    ).format(n_events_train, n_events_train/(n_events_train+n_events_test)*100., train_shift,
        n_events_test, n_events_test/(n_events_train+n_events_test)*100.))
    # Return the final `events_train` & `events_test`.
    return events_train, events_test

# def refer_events func
def refer_events(events, events_refer, path_stimulus):
    """
    Split [train,validation,test]-events according to reference events.
    :param events: `pd.DataFrame` - The loaded events.
    :param events_refer: tuple - The reference [train,test]-events.
    :param path_stimulus: path - The path of stimulus.
    :return events_train: (n_train[list],) - The list of train set, a 3s-segment starting from onset.
    :return events_test: (n_test[list],) - The list of test set, a 3s-segment starting from 500ms before onset.
    """
    # Initialize events_refer_train & events_refer_test.
    events_refer_train, events_refer_test = events_refer

    ## Prepare for splitting data.
    # Convert events from `pd.DataFrame` to `DotDict`.
    # events - (n_events[list],)
    events_ = []
    for event_idx, event_i in events.iterrows():
        events_.append(DotDict({
            "onset": event_i["onset"],
            "duration": event_i["duration"],
            "trial_type": eval(event_i["trial_type"]),
            "value": event_i["value"],
            "sample": event_i["sample"],
        }))
    events = events_
    # Construct words from events, separated by audio slices.
    events_words = []; events_words_i = []
    durations_sound = []; start_sound = 0.; stop_sound = 0.
    for event_i in events:
        if event_i.trial_type.kind == "sound":
            events_words.append(events_words_i); events_words_i = []
            durations_sound.append([start_sound, stop_sound]); start_sound = np.round(event_i.onset, decimals=3)
        elif event_i.trial_type.kind == "word":
            events_words_i.append(DotDict({
                "story": event_i.trial_type.story,
                "story_piece": int(event_i.trial_type.sound_id),
                "start": np.round(event_i.trial_type.start, decimals=3),
                "onset": np.round(event_i.onset, decimals=3),
                "word": event_i.trial_type.word,
            })); stop_sound = np.round(event_i.onset + event_i.duration, decimals=3)
    events_words.append(events_words_i); durations_sound.append([start_sound, stop_sound])
    events_words = [events_words_i for events_words_i in events_words if len(events_words_i) > 0]
    events_words = [event_word_i for events_words_i in events_words for event_word_i in events_words_i]
    durations_sound = [duration_sound_i for duration_sound_i in durations_sound if duration_sound_i[1]-duration_sound_i[0] > 0.]
    story_type = events_words[0].story
    # Check whether all items in events_refer_test are the same with events_words.
    # And in the meanwhile, we reconstruct events_test from events_refer_test.
    scale_start = 0
    scale_story_piece = np.max([duration_sound_i[1] - duration_sound_i[0] for duration_sound_i in durations_sound])
    assert scale_story_piece > 0.
    scale_story_piece = int(np.floor(np.log10(np.abs(scale_story_piece)))) + 1 + scale_start
    scale_story = len(set([event_word_i.story_piece for event_word_i in events_words]))
    assert scale_story > 0.
    scale_story = int(np.floor(np.log10(np.abs(scale_story)))) + 1 + scale_story_piece
    events_words.sort(key=lambda x: x.story_piece*pow(10,scale_story_piece)+x.start*pow(10,scale_start))
    events_refer_test.sort(key=lambda x: x.story_piece*pow(10,scale_story_piece)+x.start*pow(10,scale_start))
    event_refer_test_idx = event_word_idx = 0; events_test = []
    while event_word_idx < len(events_words) and event_refer_test_idx < len(events_refer_test):
        event_word_i = events_words[event_word_idx]
        event_refer_test_i = events_refer_test[event_refer_test_idx]
        if event_word_i.story == event_refer_test_i.story and\
           event_word_i.story_piece == event_refer_test_i.story_piece and\
           event_word_i.start == event_refer_test_i.start and\
           event_word_i.word == event_refer_test_i.description.split("-")[1]:
            event_refer_test_idx += 1
            events_test.append(DotDict({
                "story": event_word_i.story,
                "story_piece": event_word_i.story_piece,
                "start": event_word_i.start,
                "onset": event_word_i.onset,
                "description": "-".join(["test", event_word_i.word]),
            }))
        event_word_idx += 1
    assert event_refer_test_idx == len(events_refer_test)
    # Split events_test into different story_piece.
    events_test_ = [[] for _ in durations_sound]
    for event_test_i in events_test:
        # Note: We donot care about whether the end of test segment [-0.5,2.5) is in the duration_sound.
        # TODO: Allow the end of train segment [0.0, 3.0) not in the duration_sound, then fill with 0s.
        sound_mask_i = np.array([duration_sound_i[0] <= event_test_i.onset and\
            event_test_i.onset <= duration_sound_i[1] for duration_sound_i in durations_sound], dtype=np.bool_)
        assert np.sum(sound_mask_i) == 1
        sound_idx = np.argmax(sound_mask_i); onset_sound_i = durations_sound[sound_idx][0]
        events_test_[sound_idx].append(event_test_i)
    events_test = events_test_

    ## Split train-set according to events_test.
    # Use test-set to infer the duration of train-set.
    freq_ = 1000; train_mask = np.zeros((int(np.round(durations_sound[-1][-1]*freq_, decimals=0)),), dtype=np.bool_)
    for duration_sound_i in durations_sound:
        for sample_idx in range(int(np.round(duration_sound_i[0]*freq_, decimals=0)),
                                int(np.round(duration_sound_i[1]*freq_, decimals=0))):
            train_mask[sample_idx] = True
    n_samples = np.sum(train_mask)
    for events_test_i in events_test:
        for event_test_i in events_test_i:
            test_mask_i = np.zeros_like(train_mask, dtype=np.bool_)
            for sample_idx in range(int(np.round((event_test_i.onset-0.5)*freq_, decimals=0)),
                                    int(np.round((event_test_i.onset+2.5)*freq_, decimals=0))):
                if sample_idx < test_mask_i.shape[0]: test_mask_i[sample_idx] = True
            train_mask = train_mask & ~test_mask_i
    n_samples_train_possible = np.sum(train_mask); n_samples_test = n_samples - n_samples_train_possible
    # Get durations_train to check whether duration is long enough (greater than 3s).
    durations_train = []
    for group_type, group_i in itertools.groupby(enumerate(train_mask), lambda x: x[1]):
        if group_type:
            indices = list(zip(*group_i))[0]; durations_train.append([indices[0], indices[-1] + 1])
    durations_train = [[
        np.round(duration_train_i[0]/freq_, decimals=3),
        np.round(duration_train_i[1]/freq_, decimals=3)
    ] for duration_train_i in durations_train]
    train_mask_reconstr = np.zeros_like(train_mask, dtype=np.bool_)
    for duration_train_i in durations_train:
        for sample_idx in range(int(np.round(duration_train_i[0]*freq_, decimals=0)),
                                int(np.round(duration_train_i[1]*freq_, decimals=0))):
            train_mask_reconstr[sample_idx] = True
    assert (train_mask == train_mask_reconstr).all()
    durations_train = [duration_train_i for duration_train_i in durations_train\
        if duration_train_i[1] - duration_train_i[0] > 3.]
    n_samples_train_available = int(np.sum([duration_train_i[1]-duration_train_i[0]\
        for duration_train_i in durations_train]) * freq_)
    print((
        "INFO: Get {:.2f}% (selected from {:.2f}%) samples available for trainset, and {:.2f}%"+
        " samples available for testset in preprocess.meg.gwilliams2022neural.session."
    ).format((n_samples_train_available/n_samples)*100.,
        (n_samples_train_possible/n_samples)*100., (n_samples_test/n_samples)*100.))
    # Construct trainset from durations_train.
    # Note: train_shift have a scale of 1e-3.
    events_refer_train.sort(key=lambda x: x.story_piece*pow(10,scale_story_piece)+x.start*pow(10,scale_start))
    onset_refer_train = np.array([event_refer_train_i.onset for event_refer_train_i in events_refer_train], dtype=np.float32)
    events_train = []; train_shift = np.round(np.min(onset_refer_train[1:] - onset_refer_train[:-1]), decimals=3)
    # Note: We assume that the start of sound is exactly with `start=0`.
    for duration_train_i in durations_train:
        onset = duration_train_i[0]
        while np.round(onset + 3., decimals=3) <= duration_train_i[1]:
            sound_mask_i = np.array([duration_sound_i[0] <= onset and np.round(onset + 3., decimals=3) <= duration_sound_i[1]\
                for duration_sound_i in durations_sound], dtype=np.bool_)
            assert np.sum(sound_mask_i) == 1
            onset_sound_i = durations_sound[np.argmax(sound_mask_i)][0]
            events_train.append(DotDict({
                "story": story_type,
                "story_piece": np.argmax(sound_mask_i),
                "onset": np.round(onset, decimals=3),
                "start": np.round(onset - onset_sound_i, decimals=3),
                "description": "-".join(["train", "segment"]),
            }))
            onset = np.round(onset + train_shift, decimals=3)
    events_test = [event_i for events_test_i in events_test for event_i in events_test_i]
    n_events_train = len(events_train); n_events_test = len(events_test)
    print((
        "INFO: Get {:d} samples ({:.2f}%) for trainset with train_shift ({:.3f}s), and {:d}"+
        " samples ({:.2f}%) for testset in preprocess.meg.gwilliams2022neural.session."
    ).format(n_events_train, n_events_train/(n_events_train+n_events_test)*100., train_shift,
        n_events_test, n_events_test/(n_events_train+n_events_test)*100.))
    # Sort events_train & events_test.
    events_train.sort(key=lambda x: x.story_piece*pow(10,scale_story_piece)+x.start*pow(10,scale_start))
    events_test.sort(key=lambda x: x.story_piece*pow(10,scale_story_piece)+x.start*pow(10,scale_start))
    # Check whether events_train are the same with events_refer_train.
    for event_train_i, event_refer_train_i in zip(events_train, events_refer_train):
        assert event_train_i.story == event_refer_train_i.story and\
               event_train_i.story_piece == event_refer_train_i.story_piece and\
               event_train_i.start == event_refer_train_i.start and\
               event_train_i.description == event_refer_train_i.description
    # Return the final `events_train` & `events_test`.
    return events_train, events_test

# def create_dataset func
def create_dataset(data, events, path_stimulus):
    """
    Create dataset from events_train & events_test.
    :param data: object - The loaded `mne.io.brainvision.brainvision.RawBrainVision` object.
    :param events: tuple - The tuple of train events (n_train[list],) and test events (n_test[list],).
    :param path_stimulus: path - The path of stimulus.
    :return dataset_train: (n_train[list],) - The list of train data items.
    :return dataset_test: (n_test[list],) - The list of test data items.
    """
    ## Prepare for creating dataset.
    # Initialize events_train & events_test.
    events_train, events_test = events
    # Get the story_type, and then get the corresponding audio data, ordered by story_piece.
    story_type = events_train[0].story
    story_pieces = set([event_train_i.story_piece for event_train_i in events_train]) |\
        set([event_test_i.story_piece for event_test_i in events_test])
    story_pieces = list(story_pieces); story_pieces.sort()
    assert (np.array(story_pieces[1:]) - np.array(story_pieces[:-1]) == 1).all()
    path_audio = os.path.join(path_stimulus, "audio")
    path_stories = [os.path.join(path_audio, "_".join([story_type.lower(), str(story_piece_i)])+".wav")\
        for story_piece_i in story_pieces]
    stories = [read(path_story_i) for path_story_i in path_stories]

    ## Get the train epochs from the original data.
    # Set the train-segment events to the original data.
    data_train = set_events(data, events_train)
    train_events, train_events_id = mne.events_from_annotations(data_train, verbose="WARNING")
    # Check whether the original events is consistent with the events of data_train.
    # This ensures that the onset sequence is aligned with that of data_train.
    # Due to the resample process of data, the onset may not be perfectly aligned.
    events_train_ = get_events(data_train)
    onset_train = np.array([event_train_i.onset for event_train_i in events_train], dtype=np.float32)
    onset_train_ = np.array([event_train_i.onset for event_train_i in events_train_], dtype=np.float32)
    assert (np.abs(onset_train - onset_train_) < 1./(2.*data_train.info["sfreq"])).all()
    # Get the train_epochs from train_events.
    train_markers = bidict(train_events_id)
    # Note: We donot do baseline-correction here, we do it latter.
    train_epochs = mne.Epochs(data_train, train_events, tmin=0., tmax=3.-(1./data_train.info["sfreq"]),
        baseline=(0., 0.), event_id=train_events_id, preload=True, verbose="WARNING")
    assert len(train_epochs.get_data()) == len(events_train)
    assert len(set([train_epoch_i.size for train_epoch_i in train_epochs.get_data()])) == 1
    train_epochs = [DotDict({
        "name":train_markers.inverse[train_event_i],
        "data":[train_epochs.info["sfreq"], train_epoch_i.T],
    }) for train_event_i, train_epoch_i in zip(train_events[:,-1], train_epochs.get_data())]
    # Create dataset_train from train_epochs.
    dataset_train = []
    for train_idx, train_epoch_i in enumerate(train_epochs):
        assert story_type == events_train[train_idx].story
        story_piece = events_train[train_idx].story_piece
        # Make sure the truncate_range is the same with the meg data.
        truncate_range = [events_train[train_idx].start, events_train[train_idx].start + 3.]
        dataset_train.append(DotDict({
            "name": train_epoch_i.name,
            "audio": truncate_audio(stories[story_piece], truncate_range),
            "data": train_epoch_i.data,
        }))
        # Check whether both data and audio have integer duration (s).
        assert dataset_train[-1].data[1].shape[0] % dataset_train[-1].data[0] == 0
        assert dataset_train[-1].audio[1].shape[0] % dataset_train[-1].audio[0] == 0

    ## Get the test epochs from the original data.
    # Set the test-segment events to the original data.
    data_test = set_events(data, events_test)
    test_events, test_events_id = mne.events_from_annotations(data_test, verbose="WARNING")
    # Check whether the original events is consistent with the events of data_test.
    # This ensures that the onset sequence is aligned with that of data_test.
    # Due to the resample process of data, the onset may not be perfectly aligned.
    events_test_ = get_events(data_test)
    onset_test = np.array([event_test_i.onset for event_test_i in events_test], dtype=np.float32)
    onset_test_ = np.array([event_test_i.onset for event_test_i in events_test_], dtype=np.float32)
    assert (np.abs(onset_test - onset_test_) < 1./(2.*data_test.info["sfreq"])).all()
    # Get the test_epochs from test_events.
    test_markers = bidict(test_events_id)
    # Note: We donot do baseline-correction here, we do it latter.
    test_epochs = mne.Epochs(data_test, test_events, tmin=-0.5, tmax=2.5-(1./data_test.info["sfreq"]),
        baseline=(0., 0.), event_id=test_events_id, preload=True, verbose="WARNING")
    assert len(test_epochs.get_data()) == len(events_test)
    assert len(set([test_epoch_i.size for test_epoch_i in test_epochs.get_data()])) == 1
    test_epochs = [DotDict({
        "name":test_markers.inverse[test_event_i],
        "data":[test_epochs.info["sfreq"], test_epoch_i.T],
    }) for test_event_i, test_epoch_i in zip(test_events[:,-1], test_epochs.get_data())]
    # Create dataset_test from test_epochs.
    dataset_test = []
    for test_idx, test_epoch_i in enumerate(test_epochs):
        assert story_type == events_test[test_idx].story
        story_piece = events_test[test_idx].story_piece
        # Make sure the truncate_range is the same with the meg data.
        truncate_range = [events_test[test_idx].start - 0.5, events_test[test_idx].start + 2.5]
        dataset_test.append(DotDict({
            "name": test_epoch_i.name,
            "audio": truncate_audio(stories[story_piece], truncate_range),
            "data": test_epoch_i.data,
        }))
        # Check whether both data and audio have integer duration (s).
        assert dataset_test[-1].data[1].shape[0] % dataset_test[-1].data[0] == 0
        assert dataset_test[-1].audio[1].shape[0] % dataset_test[-1].audio[0] == 0

    # Return the final `dataset_train` & `dataset_test`.
    return dataset_train, dataset_test

# def preprocess_dataset func
def preprocess_dataset(dataset):
    """
    Execute further preprocess over the dataset.
    :param dataset: (n_dataset[list],) - The created dataset, only with resample preprocess.
    :return dataset: (n_dataset[list],) - The preprocessed dataset.
    """
    # Preprocess each data item.
    n_clamps = []; clamp_ratio = []
    for data_idx in range(len(dataset)):
        # Execute baseline-correction.
        baseline_range = [0, int(np.round(0.5*dataset[data_idx].data[0], decimals=0))]
        baseline = dataset[data_idx].data[1][baseline_range[0]:baseline_range[1],:]
        baseline = np.mean(baseline, axis=0, keepdims=True)
        dataset[data_idx].data[1] = dataset[data_idx].data[1] - baseline
        assert (np.mean(dataset[data_idx].data[1][baseline_range[0]:baseline_range[1],:], axis=0) < 1e-12).all()
        # Use `sklearn.preprocessing.robust_scale` to normalize data.
        # Here are the arguments of `sklearn.preprocessing.robust_scale` function:
        # >>> X: {array-like, sprase matrix} of shape (n_sample, n_features)
        # >>>   - The data to center and scale. Here, `X` has the same shape with `dataset[data_idx].data[1]`.
        # >>> axis: int, default=0
        # >>>   - Axis used to compute the medians and IQR along. If 0, independently scale each feature,
        # >>>     otherwise (if 1) scale each sample. Here, we set `axis` to 0, as `X` has the same shape.
        # >>> with_centering: bool, default=True
        # >>>   - If True, center the data before scaling. Here, we use the default value.
        # >>> with_scaling: bool, default=True
        # >>>   - If True, scale the data to unit variance (or equivalently, unit standard deviation).
        # >>>     Here, we use the default value, which is aligned with the preprocess in the original paper.
        # >>>     In A.2.1., the authors write the following sentence:
        # >>>       As explained in Section 3, we first use a quantile based robust scaler such that the
        # >>>       range [-1,1] maps to the [0.25,0.75] quantile range.
        # >>> quantile_range: tuple (q_min, q_max), 0.0 < q_min < q_max < 100.0, default=(25.0, 75.0)
        # >>>   - Quantile range used to calculate scale_. By default this is equal to the IQR, i.e., q_min
        # >>>     is the first quantile and q_max is the third quantile. Here, we use the default value,
        # >>>     which is aligned with the preprocess in the original paper. In A.2.1., the authors write
        # >>>     the following sentence:
        # >>>       As explained in Section 3, we first use a quantile based robust scaler such that the
        # >>>       range [-1,1] maps to the [0.25,0.75] quantile range.
        # >>> copy: bool, default=True
        # >>>   - Set to False to perform inplace row normalization and avoid a copy (if the input
        # >>>     is already a numpy array or a scipy.sparse CSR matrix and if axis is 1). Here, we use the
        # >>>     default value.
        # >>> unit_variance: bool, default=False
        # >>>   - If True, scale data so that normally distributed features have a variance of 1. In general,
        # >>>     if the difference between the x-values of q_max and q_min for a standard normal distribution
        # >>>     is greater than 1, the dataset will be scaled down. If less than 1, the dataset will be scaled up.
        # >>>     Here, we use the default value. As in the original paper, the range [-1,1] exactly maps to
        # >>>     the [0.25,0.75] quantile range. Enabling `unit_variance` will cause that not all points
        # >>>     in the [0.25,0.75] quantile range are in [-1,1] range.
        dataset[data_idx].data[1] = robust_scale(dataset[data_idx].data[1], axis=0,
            with_centering=True, with_scaling=True, quantile_range=(25.0, 75.0), unit_variance=False)
        # Clamp values greater than 20 after normalization to minimize the impact of large outlier samples.
        clamp_range = [-20., 20.]
        n_clamps.append(np.sum(dataset[data_idx].data[1] < clamp_range[0]) +\
            np.sum(dataset[data_idx].data[1] > clamp_range[1]))
        clamp_ratio.append(n_clamps[-1] / dataset[data_idx].data[1].size)
        dataset[data_idx].data[1] = np.where(
            dataset[data_idx].data[1] < clamp_range[0],
            clamp_range[0], dataset[data_idx].data[1])
        dataset[data_idx].data[1] = np.where(
            dataset[data_idx].data[1] > clamp_range[1],
            clamp_range[1], dataset[data_idx].data[1])
    # Log information related to current preprocess.
    print((
        "INFO: Finish the preprocess of dataset with {:d} clamped values ({:.2f}%)"+
        " in preprocess.meg.gwilliams2022neural.session."
    ).format(np.sum(n_clamps), np.mean(clamp_ratio)*100.))
    # Return the final `dataset`.
    return dataset

# def truncate_audio func
def truncate_audio(audio, truncate_range):
    """
    Truncate audio piece according to truncate_range.
    :param audio: tuple - The loaded audio data using `scipy.io.wavfile.read`. The first item is
        the sample rate of audio, and the second item is the sample sequence of audio.
    :param truncate_range: tuple - The range of current truncate process.
    :return audio_truncated: tuple - The truncated audio data. The first item is the same with
        the sample rate of audio, and the second item is the truncated sequence of audio.
    """
    # Convert truncate_range to sample indices.
    sample_rate = audio[0]; range_len = truncate_range[1] - truncate_range[0]
    truncate_range = [int(np.round(truncate_range[0] * sample_rate, decimals=0)),]
    truncate_range.append(truncate_range[0] + int(np.round(range_len * sample_rate, decimals=0)))
    # Get the corresponding audio_truncated.
    # Note: If we encounter the end of audio, we can append zeros.
    audio_truncated = [sample_rate, audio[1][truncate_range[0]:truncate_range[1]]\
        if truncate_range[1] <= audio[1].shape[0] else np.concatenate([
            audio[1][truncate_range[0]:],
            np.zeros((truncate_range[1]-audio[1].shape[0],), dtype=audio[1].dtype),
        ], axis=0)
    ]
    # Return the final `audio_truncated`.
    return audio_truncated

# def set_events func
def set_events(data, events):
    """
    Set events through annotations. Due to that `data` use annotations to store events (which are annotations
    that have `duration` equal to `1 / data.info["sfreq"]`), we have to use `set_annotations` to set events-like
    annotations to align with the original data structure. We should note that the `onset` field of `events` is
    the relative time (second) of the sample, instead of the index of the sample.
    Note: `set_annotations` will overwritte the original annotations. We should include the original events in `events`.
    :param data: object - The loaded `mne.io.brainvision.brainvision.RawBrainVision` object.
    :param events: list - The list of event, each `event` DotDict contains [onset,description].
    :return data: object - The modified `mne.io.brainvision.brainvision.RawBrainVision` object.
    """
    # Initialize `duration` as `1 / data.info["sfreq"]`.
    duration = 1. / data.info["sfreq"]
    # Initialize `annotations` using `events and `duration`.
    # Note: We use `time_as_index` and `duration` to round the original onset. For example,
    # `0.9995` may be round to `1.0`, `1.1175` may be round to `1.118`.
    onset = data.time_as_index([event_i.onset for event_i in events], use_rounding=True) * duration
    annotations = mne.Annotations(
        onset=[event_i.onset for event_i in events],
        duration=[duration for _ in events],
        description=[event_i.description for event_i in events],
        orig_time=data.annotations.orig_time
    )
    # Set events through annotations.
    data.set_annotations(annotations)
    # Return the final `data`.
    return data

# def get_events func
def get_events(data):
    """
    Get events from annotations. Due to that `data` use annotations to store events (which are annotations
    that have `duration` equal to `1 / data.info["sfreq"]`, we have to use `events_from_annotations` to get
    events from annotations to align with the original data structure.
    :param data: object - The loaded `mne.io.brainvision.brainvision.RawBrainVision` object.
    :return events: list - The list of event, each `event` DotDict contains [onset,description].
    """
    # Initialize `duration` as `1 / data.info["sfreq"]`.
    duration = 1. / data.info["sfreq"]
    # Get the events from annotations. The original `description` field of `data` doesn't contain
    # the map between description and index, it will construct index from description automatically.
    # And it seems that `Stimulus/S{:3d}` always have the lowest index, no matter how many instantiants
    # are in other kinds of events.
    events, marker = mne.events_from_annotations(data, verbose="WARNING")
    marker = bidict(marker)
    # Construct `events` from np-version `events`.
    # Note: We always use detailed description to describe each event, instead of the general event id.
    # We can extract event id from detailed description, thus avoiding re-index of stimulus event.
    events = [DotDict({"onset":event_i[0]*duration,"description":event_i[2],}) for event_i in events]
    for event_idx in range(len(events)):
        events[event_idx].onset = events[event_idx].onset.astype(np.float32)
        events[event_idx].description = marker.inverse[events[event_idx].description]
    # Return the final `events`.
    return events

# def preprocess_session func
def preprocess_session(path_session, path_stimulus, events_=None):
    """
    The whole pipeline to preprocess meg data of specified session.
    :param path_session: path - The path of specified session meg data.
    :param path_stimulus: path - The path of stimulus.
    :param events_: tuple - The reference [train,test] events.
    :return dataset: tuple - The list of [train,test] data items.
    :return events: tuple - The corresponding [train,test] events.
    """
    # Load data from specified session.
    data, positions, events = load_data(path_session)
    # Check whether reference events is None.
    if events_ is None:
        # Split events to get the [train,validation,test]-set.
        events_train, events_test = split_events(events, path_stimulus)
    else:
        # Use reference events to get the [train,validation,test]-set.
        events_train, events_test = refer_events(events, events_, path_stimulus)
    # Create dataset from events_train & events_test.
    dataset_train, dataset_test = create_dataset(data, (events_train, events_test), path_stimulus)
    # Preprocess dataset to get the final dataset.
    dataset_train = preprocess_dataset(dataset_train)
    dataset_test = preprocess_dataset(dataset_test)
    # Attach positions to dataset_train & dataset_test.
    for data_train_idx in range(len(dataset_train)):
        dataset_train[data_train_idx].chan_pos = positions
        assert positions.shape[0] == dataset_train[data_train_idx].data[1].shape[1]
    for data_test_idx in range(len(dataset_test)):
        dataset_test[data_test_idx].chan_pos = positions
        assert positions.shape[0] == dataset_test[data_test_idx].data[1].shape[1]
    # Return the final `dataset` & `events` & `positions`.
    return (dataset_train, dataset_test), (events_train, events_test)

if __name__ == "__main__":
    # macro
    base = os.path.join(os.getcwd(), os.pardir, os.pardir, os.pardir)
    path_dataset = os.path.join(base, "data", "meg.gwilliams2022neural")
    path_session1 = os.path.join(path_dataset, "sub-01", "ses-0", "meg", "sub-01_ses-0_task-1_meg.con")
    path_session2 = os.path.join(path_dataset, "sub-01", "ses-1", "meg", "sub-01_ses-1_task-1_meg.con")
    path_stimulus = os.path.join(path_dataset, "stimuli")

    # Initialize random seed.
    np.random.seed(42)

    # Preprocess the specified subject session.
    dataset1, events1 = preprocess_session(path_session1, path_stimulus)
    # Preprocess the specified subject session with reference events.
    dataset2, events2 = preprocess_session(path_session2, path_stimulus, events_=events1)

