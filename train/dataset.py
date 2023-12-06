import torch
from torch.utils.data import Dataset
import pickle


class Trainset(Dataset):
    def __init__(self):
        super(Trainset).__init__()
        with open("/root/NLP/data/meg.gwilliams2022neural/meg/transition_train_data.pkl", "rb") as file:
            self.recording = pickle.load(file)

    def __getitem__(self, index):
        data = torch.from_numpy(self.recording[index]["data"][1]).float()
        chan_pos = torch.from_numpy(self.recording[index]["chan_pos"]).float()
        subject_id = torch.tensor(self.recording[index]["subject_id"]).float()
        audio = torch.from_numpy(self.recording[index]["audio"][1]).float()

        return data, chan_pos, subject_id, audio

    def __len__(self):
        return 17000

    
class Testset(Dataset):
    def __init__(self):
        super(Testset).__init__()
        with open("/root/NLP/data/meg.gwilliams2022neural/meg/transition_test_data.pkl", "rb") as file:
            self.recording = pickle.load(file)

    def __getitem__(self, index):
        data = torch.from_numpy(self.recording[index]["data"][1]).float()
        chan_pos = torch.from_numpy(self.recording[index]["chan_pos"]).float()
        subject_id = torch.tensor(self.recording[index]["subject_id"]).float()
        audio = torch.from_numpy(self.recording[index]["audio"][1]).float()
        
        return data, chan_pos, subject_id, audio
    
    def __len__(self):
        return 8000
