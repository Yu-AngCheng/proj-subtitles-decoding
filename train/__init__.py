## import train models
#import logistic_regression
# from .logistic_regression import logistic_regression
from . import logistic_regression
# import cnn_bgru
# from .naive_cnn import naive_cnn
from .import naive_cnn
# import defossez2022decoding
from . import defossez2022decoding

# set models
models = [model for model in dir() if not model.startswith("_")]

