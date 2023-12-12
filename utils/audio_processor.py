import torchaudio
from transformers import Wav2Vec2Processor


class AudioProcessor:
    def __init__(self, target_sample_rate=16000, max_audio_length=64000):
        self.target_sample_rate = target_sample_rate
        self.max_audio_length = max_audio_length

        model_name = "jonatasgrosman/wav2vec2-large-xlsr-53-english"
        self.processor = Wav2Vec2Processor.from_pretrained(model_name)

    def __call__(self, file_path):
        """
        Load the audio file, resample it to the target sample rate, and pad it or truncate it to the max_audio_length

        Parameters:
        - file_path (str): path to the audio file

        Returns:
        - padded_audio (torch.Tensor): A (1, max_audio_length) tensor containing the padded audio data
        """
        audio_data = self._load_and_resample(file_path)
        padded_audio = self._pad(audio_data)
        return padded_audio

    def _load_and_resample(self, file_path):
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
        if sample_rate != self.target_sample_rate:
            resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=self.target_sample_rate)
            waveform = resampler(waveform)
        return waveform.numpy()

    def _pad(self, audio_data):
        """
        Pad the audio data to the max_audio_length

        Parameters:
        - audio_data (np.ndarray): A (audio_length) numpy array containing the audio data

        Returns:
        - padded_audio (torch.Tensor): A (max_audio_length) tensor containing the padded audio data
        """
        padded_audio = self.processor(audio_data, padding="max_length", max_length=self.max_audio_length,
                                      return_tensors="pt", sampling_rate=16000, truncation=True).input_values[0]
        return padded_audio


if __name__ == "__main__":
    import torch

    target_sample_rate = 16000
    max_audio_length = 64000
    audio_processor = AudioProcessor(target_sample_rate=target_sample_rate, max_audio_length=max_audio_length)

    audio_files = [r"D:\audio_1-3seconds\audio_1-3seconds\segment_1_00-01-24,935_00-01-26,111.mp3",
                   r"D:\audio_1-3seconds\audio_1-3seconds\segment_2_00-01-26,996_00-01-28,728.mp3"]

    audio_processor = AudioProcessor()
    audio_data = [audio_processor(audio_file) for audio_file in audio_files]
    audio_data = torch.stack(audio_data)
    print(audio_data.shape)
