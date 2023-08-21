# This work try to reproduce Défossez's work (Decoding speech from non-invasive brain recordings)

# Dataset 
Please first load data from https://osf.io/ag3kj/, you will get like sub01/ses0/meg/...
save it to /data/meg/meg.gwilliams2022neural/
like ../data/meg.gwilliams2022neural/sub-01/ses-0/meg/sub-01_ses-0_task-0_meg.con

# Preprocess
it is structured as 
	python session.py
	python run.py
	python subject.py
	python dataset.py
But you can directly run dataset.py

# models
the defossez2022decoding is for brain_decoding model, you can find subjectlayer, spatialattention, etc. in layers file
the wav2vec is for speech_decoding model

# prarams
initialized parameters for model

# utils 
all kinds of metric, data load/save,...
Before you run the train, first get the dataset. you should run gwilliams2022neural.py in utils/data/meg

# train
Finally, run defossez2022.py please.

In brief, run dataset.py, gwilliams2022neural.py, defossez2022.py