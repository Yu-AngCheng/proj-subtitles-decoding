import torchaudio


def load_and_resample(file_path, target_sample_rate=16000):
    """
    Load the audio file and resample it to the target sample rate

    Parameters:
    - file_path (str): path to the audio file
    - target_sample_rate (int): the target sample rate

    Returns:
    - waveform (np.ndarray): the waveform of the audio file
    """
    waveform, sample_rate = torchaudio.load(file_path)

    # Convert stereo to mono
    if waveform.shape[0] == 2:
        waveform = waveform.mean(dim=0)

    # Resample if necessary
    if sample_rate != target_sample_rate:
        resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=target_sample_rate)
        waveform = resampler(waveform)
    return waveform.numpy()
