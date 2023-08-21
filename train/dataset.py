import os, sys, time
import copy as cp
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset
import pickle
# local dep
if __name__ == "__main__":
    import os, sys
    sys.path.insert(0, os.pardir)

import models, utils
from params import defossez2022decoding_params
from utils.data.meg.gwilliams2022neural import load_gwilliams2022neural_origin


class Trainset(Dataset):
    def __init__(self):
        super(Trainset).__init__()
        # self.recording = np.load('/root/NLP/data/meg/transition_train_data.npy',allow_pickle=True)
        with open("/root/NLP/data/meg.gwilliams2022neural/meg/transition_train_data.pkl", "rb") as file:
            self.recording = pickle.load(file)

    def __getitem__(self,index):
        data = torch.from_numpy(self.recording[index]["data"][1]).float()
        chan_pos = torch.from_numpy(self.recording[index]["chan_pos"]).float()
        subject_id = torch.tensor(self.recording[index]["subject_id"]).float()
        audio = torch.from_numpy(self.recording[index]["audio"][1]).float()

        # filename = '/root/NLP/data/meg.gwilliams2022neural/transition_train_data'+str(index)+'.pickle'
        # with open(filename, 'rb') as file:
        #     recording = pickle.load(file)

        # data = torch.from_numpy(recording["data"][1]).float()
        # chan_pos = torch.from_numpy(recording["chan_pos"]).float()
        # subject_id = torch.tensor(recording["subject_id"]).float()
        # audio = torch.from_numpy(recording["audio"][1]).float()
        return data, chan_pos, subject_id, audio

    def __len__(self):
        return 17000
    
class Testset(Dataset):
    def __init__(self):
        super(Testset).__init__()
        # self.recording = np.load('/root/NLP/data/meg/transition_test_data.npy',allow_pickle=True)
        with open("/root/NLP/data/meg.gwilliams2022neural/meg/transition_test_data.pkl", "rb") as file:
            self.recording = pickle.load(file)

    def __getitem__(self,index):
        data = torch.from_numpy(self.recording[index]["data"][1]).float()
        chan_pos = torch.from_numpy(self.recording[index]["chan_pos"]).float()
        subject_id = torch.tensor(self.recording[index]["subject_id"]).float()
        audio = torch.from_numpy(self.recording[index]["audio"][1]).float()
        
        # filename = '/root/NLP/data/meg.gwilliams2022neural/transition_test_data'+str(index)+'.pickle'
        # with open(filename, 'rb') as file:
        #     recording = pickle.load(file)

        # data = torch.from_numpy(recording["data"][1]).float()
        # chan_pos = torch.from_numpy(recording["chan_pos"]).float()
        # subject_id = torch.tensor(recording["subject_id"]).float()
        # audio = torch.from_numpy(recording["audio"][1]).float()
        return data, chan_pos, subject_id, audio
    
    def __len__(self):
        return 8000