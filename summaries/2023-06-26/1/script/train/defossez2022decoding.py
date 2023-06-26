import os, sys, time
import copy as cp
import numpy as np
import tensorflow as tf
# local dep
if __name__ == "__main__":
    import os, sys
    sys.path.insert(0, os.pardir)
import models, utils

__all__ = [
    "train",
]

# Global variables.
params = None; paths = None
model = None; optimizer = None
dataset_train = None; dataset_test = None
outputs = None; loss = None; accuracy = None; time_start = None; time_stop = None

"""
init funcs
"""
# def init func
def init(base_, params_):
    """
    Initialize `defossez2022decoding` training variables.
    :param base_: The base path of current project.
    :param params_: The parameters of current training process.
    """
    global params, paths
    # Initialize params.
    params = cp.deepcopy(params_)
    paths = utils.Paths.Paths(base=base_, params=params)
    # Initialize model.
    _init_model()
    # Initialize training process.
    _init_train()

# def _init_model func
def _init_model():
    """
    Initialize model used in the training process.
    """
    global params, model, optimizer
    ## Initialize tf configuration.
    # Not set random seed, should be done before instantiating `model`.
    # Set default precision.
    tf.keras.backend.set_floatx(params._precision)
    # Check whether run in graph mode or eager mode.
    tf.config.run_functions_eagerly(not params.train.use_graph_mode)

    ## Initialize model.
    # Only `params.model` is consistent with `model.params`.
    model = models.defossez2022decoding(params.model)
    # Make an ADAM optimizer for model.
    optimizer = tf.keras.optimizers.Adam(learning_rate=params.train.lr_i)

# def _init_train func
def _init_train():
    """
    Initialize the training process.
    """
    pass

"""
data funcs
"""
# def load_data func
def load_data():
    """
    Load data from specified dataset.
    :return dataset_train_: tf.data.Dataset - The input train dataset.
    :return dataset_test_: tf.data.Dataset - The input test dataset.
    """
    global params
    # Load data from specified dataset.
    try:
        func = getattr(sys.modules[__name__], "_".join(["_load_data", params.train.dataset]))
        dataset_train_, dataset_test_ = func()
    except Exception:
        raise ValueError("ERROR: Unknown dataset type {} in train.defossez2022decoding.".format(params.train.dataset))
    # Return the final `dataset_train_` & `dataset_test_`.
    return dataset_train_, dataset_test_

# def _load_data_mnist func
def _load_data_mnist():
    """
    Load data from `mnist` dataset.
    :return dataset_train_: tf.data.Dataset - The input train dataset.
    :return dataset_test_: tf.data.Dataset - The input test dataset.
    """
    global params, paths
    # Load data from `mnist` dataset.
    # train_data - (n_train, 28, 28), train_label - (n_train,)
    # test_data - (n_test, 28, 28), train_label - (n_test,)
    (train_data, train_label), (test_data, test_label) = tf.keras.datasets.mnist.load_data()
    # Create random locations, subject_id.
    # locations - (28, 2), subject_id - ()
    locations = np.random.uniform(size=(train_data.shape[-1], 2)).astype(np.float32)
    assert np.min(locations) >= 0. and np.max(locations) <= 1.
    subject_id = np.squeeze(np.random.poisson(lam=1, size=(1,))).astype(np.float32) + 1
    subject_id = subject_id / np.max(subject_id); assert np.min(subject_id) > 0. and np.max(subject_id) <= 1.
    # Construct dataset from mnist dataset.
    # locations_train - (n_train, 28, 2), subject_id_train - (n_train,)
    X_train = train_data.astype(np.float32); y_train = np.eye(len(set(np.squeeze(train_label))))[train_label].astype(np.float32)
    locations_train = np.stack([locations for _ in range(X_train.shape[0])], axis=0)
    subject_id_train = np.stack([subject_id for _ in range(X_train.shape[0])], axis=0)
    dataset_train_ = tf.data.Dataset.from_tensor_slices((X_train, locations_train, subject_id_train, y_train))
    # locations_test - (n_test, 28, 2), subject_id_test - (n_test,)
    X_test = test_data.astype(np.float32); y_test = np.eye(len(set(np.squeeze(test_label))))[test_label].astype(np.float32)
    locations_test = np.stack([locations for _ in range(X_test.shape[0])], axis=0)
    subject_id_test = np.stack([subject_id for _ in range(X_test.shape[0])], axis=0)
    dataset_test_ = tf.data.Dataset.from_tensor_slices((X_test, locations_test, subject_id_test, y_test))
    # Shuffle and then batch the dataset.
    dataset_train_ = dataset_train_.shuffle(params.train.buffer_size).batch(params.train.batch_size, drop_remainder=True)
    dataset_test_ = dataset_test_.shuffle(params.train.buffer_size).batch(params.train.batch_size, drop_remainder=True)
    # Return the final `dataset_train_` & `dataset_test_`.
    return dataset_train_, dataset_test_

# def _load_data_meg_liu2019cell func
def _load_data_meg_liu2019cell():
    """
    Load data from `meg_liu2019cell` dataset.
    :return dataset_train_: tf.data.Dataset - The input train dataset.
    :return dataset_test_: tf.data.Dataset - The input test dataset.
    """
    global params, paths
    # Load data from `meg_liu2019cell` dataset.
    # data - (n_samples[list],)
    data = utils.data.meg.meg_liu2019cell.load_meg_liu2019cell(
        path_data=os.path.join(paths.base, "data", "meg.liu2019cell"), subjects_allowed=None
    ); np.random.shuffle(data)
    # X - (n_samples, seq_len, n_channels)
    X = np.array([data_i.data for data_i in data], dtype=np.float32)[:,1800:,:]
    # locations - (n_samples, n_channels, 2)
    locations = np.array([data_i.channel_locations for data_i in data], dtype=np.float32)[:,:,:2]
    assert np.min(locations) >= 0. and np.max(locations) <= 1.
    # subject_id - (n_samples, n_subjects)
    subject_id = np.array([data_i.subject_id for data_i in data], dtype=np.int64)
    n_subjects = len(set(subject_id)); subject_id = np.eye(n_subjects)[subject_id].astype(np.float32)
    # y - (n_samples, n_labels)
    labels = list(set([data_i.label for data_i in data])); labels.sort(); assert len(labels) == 8
    y = np.array([labels.index(data_i.label) for data_i in data], dtype=np.int64)
    y = np.eye(len(labels))[y]
    # Construct dataset from data items.
    train_ratio = params.train.train_ratio
    X_train = X[:int(train_ratio*X.shape[0]),:,:]; locations_train = locations[:int(train_ratio*locations.shape[0]),:,:]
    subject_id_train = subject_id[:int(train_ratio*subject_id.shape[0])]; y_train = y[:int(train_ratio*y.shape[0]),:]
    X_test = X[int(train_ratio*X.shape[0]):,:,:]; locations_test = locations[int(train_ratio*locations.shape[0]):,:,:]
    subject_id_test = subject_id[int(train_ratio*subject_id.shape[0]):]; y_test = y[int(train_ratio*y.shape[0]):,:]
    dataset_train_ = tf.data.Dataset.from_tensor_slices((X_train, locations_train, subject_id_train, y_train))
    dataset_test_ = tf.data.Dataset.from_tensor_slices((X_test, locations_test, subject_id_test, y_test))
    # Shuffle and then batch the dataset.
    dataset_train_ = dataset_train_.shuffle(params.train.buffer_size).batch(params.train.batch_size, drop_remainder=True)
    dataset_test_ = dataset_test_.shuffle(params.train.buffer_size).batch(params.train.batch_size, drop_remainder=True)
    # Return the final `dataset_train_` & `dataset_test_`.
    return dataset_train_, dataset_test_

"""
train funcs
"""
# def train func
def train(base_, params_):
    """
    Train the model.
    :param base_: The base path of current project.
    :param params_: The parameters of current training process.
    """
    global params, paths, model, optimizer, dataset_train, dataset_test
    global outputs, loss, accuracy, time_start, time_stop
    # Initialize parameters & variables of current training process.
    init(base_, params_)
    # Load data from specified dataset.
    dataset_train, dataset_test = load_data()
    # Start training process.
    paths.run.logger.summaries.info("Training Started.")
    # Train the model for `params.train.n_epochs` epochs.
    iter_idx = 0
    for epoch_idx in range(params.train.n_epochs):
        # Start training epoch.
        paths.run.logger.summaries.info("Start training epoch {:d}.".format(epoch_idx))

        for train_batch in dataset_train:
            # Record the start time of preparing data.
            time_start = time.time()
            # Prepare parameters for current iteration.
            params.iteration(iteration=iter_idx)
            optimizer.lr = params.train.lr_i

            # Execute model for current iteration.
            outputs, loss = _train(train_batch); outputs, loss = outputs.numpy(), loss.numpy()
            accuracy = np.argmax(outputs, axis=-1) == np.argmax(train_batch[-1], axis=-1)
            accuracy = np.sum(accuracy) / accuracy.size
            # Record current training iteration.
            time_stop = time.time()
            print((
                "Finish train iter {:d} in {:.2f} seconds, generating {:d} concrete functions."
            ).format(iter_idx, time_stop-time_start, len(_train.pretty_printed_concrete_signatures().split("\n\n"))))

            # Log & test & save current iteration, then update iter_idx.
            _log(iter_idx); _test(iter_idx); _save(iter_idx); iter_idx += 1

        # Finish training epoch.
        paths.run.logger.summaries.info("Finished training epoch {:d}.".format(epoch_idx))
    # Finish training process.
    paths.run.logger.summaries.info("Training Finished.")

# def _forward func
@tf.function
def _forward(inputs, training=False):
    """
    Forward the model using one-step data. Everything entering this function already be a tensor.
    :param inputs: (X, locations, subject_id, y)
    :param training: Indicate whether enable training process.
    :return outputs_: (n_samples, n_labels) - The predicted labels of inputs.
    :return loss_: float - The corresponding cross-entropy loss.
    """
    global model; return model(inputs, training=training)

# def _train func
@tf.function
def _train(inputs):
    """
    Train the model using one-step data. Everything entering this function already be a tensor.
    :param inputs: (X, locations, subject_id, y)
    :return outputs_: (n_samples, n_labels) - The predicted labels of inputs.
    :return loss_: float - The corresponding cross-entropy loss.
    """
    global model, optimizer
    # Train the model using one-step data.
    with tf.GradientTape() as gt:
        outputs_, loss_ = _forward(inputs, training=True)
    # Modify weights to optimize the model.
    gradients = gt.gradient(loss_, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    # Return the final `outputs_` & `loss_`.
    return outputs_, loss_

# def _log func
def _log(iteration):
    """
    Log the information of current iteration.
    :param iteration: The index of current iteration.
    """
    global params, paths, outputs, loss, accuracy, time_start, time_stop
    if iteration % params.train.i_log == 0:
        ## Write series of messages to logger from this backprop iteration.
        # Run time of current training process.
        msg = "Finished backprop iter {:d} in {:.2f} seconds.".format(iteration, time_stop-time_start)
        paths.run.logger.summaries.info(msg)
        # Run [accuracy,loss] of current test process.
        msg = "Accuracy(train): {:.2f}%. Loss(train): {:.5f}.".format(accuracy*100., loss)
        paths.run.logger.summaries.info(msg)
        # Iteration parameters of current training process.
        msg = "Iteration Parameters:"
        iter_params, keys = utils.DotDict(), utils.DotDict.iter_keys(params)
        for key in keys:
            if key[-1].endswith("_i"):
                setattr(iter_params, ".".join(key), utils.DotDict.iter_getattr(params, key))
        for key, val in iter_params.items():
            if isinstance(val, float): msg += " <{}> {:.5f}".format(key, val)
            else: msg += " <{}> {}".format(key, val)
        paths.run.logger.summaries.info(msg)

        ### Write progress to tensorboard.
        with paths.run.logger.tensorboard.as_default():
            # Run time.
            tf.summary.scalar("time", time_stop-time_start, iteration)
            # Run loss.
            tf.summary.scalar("loss", loss, step=iteration)
            # Run accuracy.
            tf.summary.scalar("accuracy", accuracy, step=iteration)
            # Iteration parameters.
            iter_params, keys = utils.DotDict(), utils.DotDict.iter_keys(params)
            for key in keys:
                if key[-1].endswith("_i"):
                    setattr(iter_params, os.path.join(*key), utils.DotDict.iter_getattr(params, key))
            for key, val in iter_params.items():
                # Temporally only support float type parameters.
                if isinstance(val, float):
                    tf.summary.scalar(os.path.join("params", key), val, step=iteration)
        paths.run.logger.tensorboard.flush()

# def _test func
def _test(iteration):
    """
    Test the model using `dataset_test`.
    :param iteration: The index of current training iteration.
    """
    global params, paths, model, dataset_test
    if iteration % params.train.i_test == 0:
        ## Run current test process.
        # Record the start time of current test process.
        time_start_ = time.time()
        # Execute test process.
        accuracy_, loss_ = [], []
        for test_batch in dataset_test:
            outputs_i, loss_i = _forward(test_batch)
            outputs_i, loss_i = outputs_i.numpy(), loss_i.numpy()
            accuracy_i = np.argmax(outputs_i, axis=-1) == np.argmax(test_batch[-1], axis=-1)
            accuracy_i = np.sum(accuracy_i) / accuracy_i.size
            accuracy_.append(accuracy_i); loss_.append(loss_i)
        accuracy_ = np.mean(accuracy_); loss_ = np.mean(loss_)
        # Record the end time of current test process.
        time_stop_ = time.time()

        ## Log current test process.
        # Run time of current test process.
        msg = "Finished test iter {:d} in {:.2f} seconds.".format(iteration, time_stop_-time_start_)
        paths.run.logger.summaries.info(msg)
        # Run [accuracy,loss] of current test process.
        msg = "Accuracy(test): {:.2f}%. Loss(test): {:.5f}.".format(accuracy_*100., loss_)
        paths.run.logger.summaries.info(msg)

# def _save func
def _save(iteration):
    """
    Save model of current iteration.
    :param iteration: The index of current iteration.
    """
    global params, paths, model
    if iteration % params.train.i_model == 0:
        # Save the model of current iteration.
        model.save_weights(os.path.join(paths.run.model, str(iteration)))

if __name__ == "__main__":
    import os
    # local dep
    from params import defossez2022decoding_params

    # macro
    dataset = "gwilliams2022neural"
    n_channels = 273; n_features = 128; n_labels = 8
    # if dataset == "meg_liu2019cell":
    #     n_channels = 273; n_features = 128; n_labels = 8
    # elif dataset == "mnist":
    #     n_channels = 28; n_features = 128; n_labels = 10

    # Initialize random seed.
    utils.model.set_seeds(42)

    ## Instantiate defossez2022decoding.
    # Initialize base.
    base = os.path.join(os.getcwd(), os.pardir)
    # Instantiate defossez2022decoding_params.
    d2d_params_inst = defossez2022decoding_params(n_channels=n_channels, n_features=n_features, n_labels=n_labels)
    d2d_params_inst.train.n_epochs = 50; d2d_params_inst.train.batch_size = 128; d2d_params_inst.train.dataset = dataset
    d2d_params_inst.train.i_log = 10; d2d_params_inst.train.i_test = 100; d2d_params_inst.train.i_model = 1000
    # Train defossez2022decoding.
    train(base, d2d_params_inst)

