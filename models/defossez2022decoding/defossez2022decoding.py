import torch
import torch.nn as nn
from models.defossez2022decoding.layers.Conv1DBlock import Conv1DBlock
from models.defossez2022decoding.layers.SubjectBlock import SubjectBlock

import utils

__all__ = [
    "defossez2022decoding",
]

class defossez2022decoding(nn.Module):
    def __init__(self, params):
        super(defossez2022decoding, self).__init__()
        self.params = params
        self.n_subjects = params.n_subjects
        self.subject_block = SubjectBlock(self.n_subjects,[208, 208, 320], 3, 0.3)
        self.conv1d_block = nn.Sequential(
            Conv1DBlock(320,[320, 320, 640], [3, 3, 3], [1, 2, 2]),
            Conv1DBlock(320,[320, 320, 640], [3, 3, 3], [4, 8, 2]),
            Conv1DBlock(320,[320, 320, 640], [3, 3, 3], [16, 1, 2]),
            Conv1DBlock(320,[320, 320, 640], [3, 3, 3], [2, 4, 2]),
            Conv1DBlock(320,[320, 320, 640], [3, 3, 3], [8, 16, 2])
        )
        self.feature_block = nn.Sequential(
            nn.Conv1d(320, 640, kernel_size=1),
            nn.GELU(),
            nn.Conv1d(640, self.params.n_features, kernel_size=1)
        )
        self.mlp_feature = nn.Sequential(
            nn.Linear(self.params.n_features, self.params.n_features),
            nn.GELU(),
            nn.Linear(self.params.n_features, self.params.n_features),
        )
        # self.feature_time = nn.Sequential(
        #     nn.Linear(360, 149),
        #     nn.GELU(),
        #     nn.Linear(149, 149),
        # )
        self.feature_time = nn.Conv1d(360, 149, kernel_size=3, stride=1, padding=1)

    def forward(self, inputs):
        X, locations, subject_id = inputs
        outputs = self.subject_block((X, locations, subject_id))
        outputs = self.conv1d_block(outputs)
        outputs = self.feature_block(outputs.permute(0, 2, 1)) # (batch_size, n_features, seq_len)
        outputs = self.mlp_feature(outputs.permute(0, 2, 1)) # (batch_size, seq_len, n_features)
        # outputs = self.feature_time(outputs.permute(0,2,1)) # (batch_size, n_features, seq_len）
        # return outputs.permute(0,2,1) # (batch_size, seq_len, n_features）
        outputs = self.feature_time(outputs) # (batch_size, seq_len, n_features)
        return outputs # (batch_size, seq_len, n_features）


if __name__ == "__main__":
    import numpy as np
    # local dep
    from params import defossez2022decoding_params

    batch_size = 16; seq_len = 850; n_channels = 55; n_features = 128;  n_subjects = 42

    utils.model.set_seeds(42)
    d2d_params_inst = defossez2022decoding_params(n_channels=n_channels, n_subjects=n_subjects, n_features=n_features)
    d2d_inst = defossez2022decoding(d2d_params_inst.model)
    # print(d2d_inst)
    X = torch.randn(batch_size, seq_len, n_channels)
    locations = torch.randn(batch_size, n_channels, 2)
    subject_id = torch.eye(n_subjects)[np.random.randint(0, n_subjects, size=(batch_size,))].float()
    inputs = (X, locations, subject_id)
    outputs = d2d_inst(inputs)
    
    
