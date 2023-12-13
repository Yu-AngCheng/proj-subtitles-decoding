import torch
import torchaudio
from transformers import Wav2Vec2Processor


class AudioProcessor:
    """
    A class that preprocesses the audio data. It resamples the audio to the target sample rate, and pads the audio to
    the target_max_length

    Parameters:
    - target_max_length (int): The max length of the audio data after preprocessing
    - orig_sample_rate (int): The original sample rate of the audio data
    - target_sample_rate (int): The target sample rate of the audio data
    """
    def __init__(self, target_max_length, orig_sample_rate=44100, target_sample_rate=16000):
        self.target_max_length = target_max_length
        self.target_sample_rate = target_sample_rate
        self.orig_sample_rate = orig_sample_rate

        model_name = "jonatasgrosman/wav2vec2-large-xlsr-53-english"
        self.processor = Wav2Vec2Processor.from_pretrained(model_name)

    def __call__(self, audio):
        """
        Resample the audio with the target_sample_rate, and pad it to the target_max_length

        Parameters:
        - audio (np.ndarray): A numpy array containing the audio

        Returns:
        - padded_audio (torch.Tensor): A (max_audio_length,) tensor containing the preprocessed audio
        """
        audio = self._resample(audio)
        padded_audio = self._pad(audio)
        return padded_audio

    def _resample(self, audio):
        """
        Resample the audio to the target_sample_rate

        Parameters:
        - audio (np.ndarray): A numpy array containing the audio

        Returns:
        - audio (np.ndarray): A numpy array containing the resampled audio
        """
        audio = torch.tensor(audio, dtype=torch.float32)
        resampler = torchaudio.transforms.Resample(orig_freq=self.orig_sample_rate, new_freq=self.target_sample_rate)
        audio = resampler(audio).numpy()
        return audio

    def _pad(self, audio):
        """
        Pad the audio data to the target_max_length

        Parameters:
        - audio (np.ndarray): A numpy array containing the audio

        Returns:
        - padded_audio (torch.Tensor): A (target_max_length,) tensor containing the padded audio
        """
        padded_audio = self.processor(audio, padding="max_length", max_length=self.target_max_length,
                                      return_tensors="pt", sampling_rate=16000, truncation=True).input_values[0]
        return padded_audio


if __name__ == "__main__":
    import numpy as np
    data_file = '../data/data_segmented.npy'
    data = np.load(data_file, allow_pickle=True)
    audios = data.item()['audio']
    audio = audios[0]
    print(f'The original audio data has length {audio.shape[0]}')

    target_audio_sample_rate = 16000
    orig_audio_sample_rate = 44100

    orig_max_length = max([audio.shape[0] for audio in audios])
    print(f'The original max length is {orig_max_length}')

    target_max_length = int(orig_max_length * (target_audio_sample_rate / orig_audio_sample_rate))
    print(f'The target max length is {target_max_length}')

    audio_processor = AudioProcessor(target_max_length=target_max_length, orig_sample_rate=orig_audio_sample_rate,
                                     target_sample_rate=target_audio_sample_rate)

    processed_audio = audio_processor(audio)
    print(f'The processed audio data has length {processed_audio.shape[0]}')
