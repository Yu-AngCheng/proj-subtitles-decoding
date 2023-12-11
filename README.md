# Decoding Semantic Information from sEEG Recordings in a Naturalistic Setting



## Project Overview

In this project, our team - Sixuan Chen, Laila Johnston, Leyang Hu, and Yu-Ang Cheng - aims to extend the boundaries of speech decoding by applying advanced neural network models to Stereoelectroencephalography (sEEG) data. Our goal is to decode semantic information from sEEG recordings obtained in a naturalistic setting, specifically while subjects watch a movie. This approach allows us to capture a richer and more varied set of neural responses compared to traditional methods.



## Prerequisites

Ensure the following prerequisites are met for running the code:

1. **Install PyTorch with CUDA:**

   `pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118`

2. **Install Additional Dependencies:**

   `pip install -r requirements.txt`

​	Ensure you run this command in the root directory of the project.





## Training the Model

#### Initiating Training

To initiate the training process, execute the following command:

`python -m train.train`

Ensure you run this command in the root directory of the project.



#### Customizing Training Parameters

The training script supports various parameters for customization:

- `--exp_name` (`-e`): Set the experiment name; affects where checkpoints/logs are saved. Checkpoints and logs will be saved in `/experiments/$EXP_NAME`. Default: 'lr_1e-3-batch_10-train_ratio-0.8'.
- `--lr` (`-l`): Learning rate. Default: `1e-3`.
- `--save_freq` (`-s`): Model saving frequency. Default: `1`.
- `--total_epoch` (`-t`): Total number of training epochs. Default: `20`.
- `--cont` (`-c`): Continue training from the latest checkpoint.
- `--batch_size` (`-b`): Batch size. Default: `10`.
- `--audio_dir` (`-ad`): Path to the audio data folder. Default: './data/audio_1-4seconds'.
- `--seeg_dir` (`-sd`): Path to the sEEG data folder. Default: './data/seeg_1-4seconds'.
- `--train_ratio` (`-r`): Ratio of training data to total data. Default: `0.8`.
- `--num_workers` (`-w`): Number of workers for data loading. Default: `4`.

To use these parameters, append them to the training command. For example:

`python -m train.train --lr 0.001 --batch_size 20 --train_ratio 0.75`

This example command starts training with a learning rate of 0.001, a batch size of 20, and uses 75% of the data for training.

