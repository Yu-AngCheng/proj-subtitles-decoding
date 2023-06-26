import os
import sys
import argparse
import utils, params,train


# Initialize parser.
parser = argparse.ArgumentParser(description="meg training parameters.")
parser.add_argument("-m", "--model",
    choices=train.models, default=train.models[0],
    help="The model of current training process.")

parser.add_argument("-s", "--seed", default=42,
    help="The random seed of training process.")

if __name__ == "__main__":
    ## Initialize model.
    # Initialize base.
    base = os.getcwd()
    # Initialize args.
    args = parser.parse_args()
    # Initialize random seed.
    utils.model.set_seeds(int(args.seed))
    # Initialize params.
    params_name = args.model.split("_"); params_name.append("params")
    params_name = "_".join(params_name)
    params_ = getattr(params, params_name); params_ = params_()

    ## Train model.
    train_name = args.model
    train_ = getattr(train, train_name)
    train_.train(base, params_)