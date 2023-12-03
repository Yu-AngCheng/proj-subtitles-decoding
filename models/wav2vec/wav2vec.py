import torch
import logging
import torch.nn as nn
from utils.data import save_pickle, load_pickle
from transformers import Wav2Vec2Processor, Wav2Vec2Model
from datasets import load_dataset


# write a class for wav2vec
class wav2vec(nn.Module):
    def __init__(self, model_name="jonatasgrosman/wav2vec2-large-xlsr-53-english"):
        super(wav2vec, self).__init__()
        self.model_name = model_name
        self.processor = Wav2Vec2Processor.from_pretrained(self.model_name)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = Wav2Vec2Model.from_pretrained(self.model_name).to(self.device)

        # self.logger = logging.getLogger(__name__)
        # self.logger.info("wav2vec model loaded.")
        # self.seq_feature = torch.nn.Sequential(
        #     nn.Linear(1024, 360), #768
        #     nn.GELU(),
        #     nn.Linear(360, 128),
        # )

        self.sampling_rate = 16000
        
        self.seq_feature = torch.nn.Sequential(
            nn.Conv1d(1024, 360, kernel_size=3, stride=1, padding=1),
            nn.GELU(),
            nn.Conv1d(360, 128, kernel_size=3, stride=1, padding=1),
        )


    def forward(self, audio, return_tensors="pt", padding="longest"):
        input_values = self.processor(audio,sampling_rate=16000, return_tensors=return_tensors, padding=padding).input_values.to(self.device)
        representation = self.model(input_values.squeeze())
        output = representation.last_hidden_state # batch_size, sequence_length, hidden_size= 768
        # output = self.seq_feature(output) # (batch,sequence_length, 128)
        # return output # (batch,sequence_length=149, 128)
        output = self.seq_feature(output.permute(0,2,1)) # (batch,128, seq_length = 360)
        return output.permute(0,2,1) # (batch, seq_length = 360,128)

    # def save(self, path):
    #     torch.save(self.model.state_dict(), path)
    #     self.logger.info("Wav2Vec2 model saved.")

    # def load(self, path):
    #     self.model.load_state_dict(torch.load(path, map_location=self.device))
    #     self.logger.info("Wav2Vec2 model loaded.")


if __name__ == "__main__":
    # load model and tokenizer
    processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
    model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base-960h")

    # load dummy dataset and read soundfiles
    # ds = load_dataset("patrickvonplaten/librispeech_asr_dummy", "clean", split="validation")

    # tokenize
    # input_values = processor(ds[0]["audio"]["array"], return_tensors="pt", padding="longest").input_values  # Batch size 1
    # input ds with batch size = 2
    

    print('start read file')
    file = load_pickle("F:\\PKU2\\curriculum\\computer_science\\NLP\\Hw4\\mycode\\data\\meg.gwilliams2022neural\\sub-01\\dataset.origin")
    print('read file done')

    input_values = processor(
        file['train'][0]["audio"][1], return_tensors="pt", padding="longest").input_values  # Batch size 1

    # get representation
    representation = model(input_values)
    print(representation.shape)