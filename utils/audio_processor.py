import torch
import torchaudio
from transformers import Wav2Vec2Processor


class AudioProcessor:
    def __init__(self, target_max_length, orig_sample_rate=44100, target_sample_rate=16000):
        self.target_max_length = target_max_length
        self.target_sample_rate = target_sample_rate
        self.orig_sample_rate = orig_sample_rate

        model_name = "jonatasgrosman/wav2vec2-large-xlsr-53-english"
        self.processor = Wav2Vec2Processor.from_pretrained(model_name)

    def __call__(self, audio):
        """
        Resample the audio, and pad it to the max_audio_length

        Parameters:
        - audio (np.ndarray): A numpy array containing the audio

        Returns:
        - padded_audio (torch.Tensor): A (max_audio_length,) tensor containing the padded audio
        """
        audio = self._resample(audio)
        padded_audio = self._pad(audio)
        return padded_audio

    def _resample(self, audio):
        """
        Resample the audio to the target sample rate

        Parameters:
        - audio (np.ndarray): A numpy array containing the audio

        Returns:
        - waveform (np.ndarray): the waveform of the audio file
        """
        audio = torch.tensor(audio, dtype=torch.float32)
        resampler = torchaudio.transforms.Resample(orig_freq=self.orig_sample_rate, new_freq=self.target_sample_rate)
        waveform = resampler(audio)
        return waveform.numpy()

    def _pad(self, audio):
        """
        Pad the audio data to the max_audio_length

        Parameters:
        - audio (np.ndarray): A numpy array containing the audio

        Returns:
        - padded_audio (torch.Tensor): A (max_audio_length) tensor containing the padded audio data
        """
        padded_audio = self.processor(audio, padding="max_length", max_length=self.target_max_length,
                                      return_tensors="pt", sampling_rate=16000).input_values[0]
        return padded_audio


if __name__ == "__main__":
    import numpy as np
    data_file = '../data/data_segmented.npy'
    data = np.load(data_file, allow_pickle=True)
    audios = data.item()['audio']
    audio = audios[0]
    print(f'The original audio data has shape {audio.shape}')

    target_audio_sample_rate = 16000
    orig_audio_sample_rate = 44100

    orig_max_length = max([audio.shape[0] for audio in audios])
    print(f'The original max length is {orig_max_length}')

    target_max_length = int(orig_max_length * (target_audio_sample_rate / orig_audio_sample_rate))
    print(f'The target max length is {target_max_length}')

    audio_processor = AudioProcessor(target_sample_rate=target_audio_sample_rate, target_max_length=target_max_length)

    processed_audio = audio_processor(audio)
    print(f'The processed audio data has shape {processed_audio.shape}')
