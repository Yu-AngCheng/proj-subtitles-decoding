import copy as cp
import numpy as np
from sklearn.linear_model import Lasso
# local dep
if __name__ == "__main__":
    import os, sys
    sys.path.insert(0, os.pardir)

__all__ = [
    "lasso_regression",
]

class lasso_regression:
    """
    `lasso_regression` classifier, without considering time information.
    """

    def __init__(self, params):
        """
        Initialize `lasso_regression` object.
        :param params: Model parameters initialized by lasso_regression_params.
        """
        # First call super class init function to set up `object`
        # style model and inherit it's functionality.
        super(lasso_regression, self).__init__()

        # Copy hyperparameters (e.g. initialized parameters) from parameter dotdict, usually
        # generated from lasso_regression_params() in params/lasso_regression_params.py.
        self.params = cp.deepcopy(params)

        # Create trainable model.
        self.model = Lasso(**self.params)

    # def fit func
    def fit(self, X, y):
        """
        Fit model with coordinate descent, instead of max-likelihood estimation.
        Note: Coordinate descent is an algorithm that considers each column of data at a time hence it will
        automatically convert the X input as a Fortran-contiguous numpy array if necessary.
        :param X: (n_samples, n_features) - The input data.
        :param y: (n_samples,) - The target data. Will be cast to X's dtype if necessary.
        """
        # Fit the model with train data.
        self.model.fit(X, y, sample_weight=None, check_input=True)

    # def predict func
    def predict(self, X):
        """
        Predict using the lasso regression model.
        :param X: (n_samples, n_features) - The input data.
        :return y_pred: (n_samples,) - The predicted target data.
        """
        # Predict using logistic regression model.
        y_pred = self.model.predict(X)
        # Return the final `y_pred`.
        return np.round(y_pred)

# def create_toy_data func
def create_toy_data(n_samples=50, n_features=2, n_classes=5):
    """
    Create toy data from specified parameters.
    :param n_samples: The number of samples corresponding to each class.
    :param n_features: The number of features.
    :param n_classes: The number of classes.
    :return X: (n_samples, n_features) - The source data.
    :return y: (n_samples,) - The target data.
    """
    assert n_samples >= 2 and n_features >= 2 and n_classes >= 2
    # Initialize offset according to `n_classes`.
    offset = np.arange(n_classes, dtype=np.float32)
    # Get the random data samples of each class.
    X, y = [], []
    for class_idx in range(len(offset)):
        X.append(np.random.random(size=(n_samples, n_features)).astype(np.float32) + offset[class_idx])
        y.append(np.array([class_idx for _ in range(n_samples)], dtype=np.float32))
    X = np.concatenate(X, axis=0); y = np.concatenate(y, axis=0)
    # Shuffle the original data.
    data = np.concatenate([X, y.reshape((-1,1))], axis=-1); np.random.shuffle(data)
    X = data[:,:X.shape[1]]; y = data[:,-1]
    # Return the final `X` & `y`.
    return X, y

if __name__ == "__main__":
    import os
    import pandas as pd
    import seaborn as sns
    import matplotlib.pyplot as plt
    # local dep
    import utils
    from params import lasso_regression_params

    # Initialize random seed.
    np.random.seed(42)
    # Initialize image path.
    path_img = os.path.join(os.getcwd(), "img")
    if not os.path.exists(path_img): os.makedirs(path_img)

    # Get the [X,y] of toy data.
    X, y = create_toy_data(n_samples=50, n_features=2, n_classes=5)
    data = pd.DataFrame(utils.DotDict({"x0":X[:,0],"x1":X[:,1],"y":y,}))
    sns.scatterplot(data=data, x="x0", y="x1", hue="y"); plt.savefig(os.path.join(path_img, "lasso_regression.data.png"))
    # Create train-set and test-set from whole data.
    X_train = X[:int(0.8*X.shape[0]),:]; y_train = y[:int(0.8*y.shape[0])]
    X_test = X[int(0.8*X.shape[0]):,:]; y_test = y[int(0.8*y.shape[0]):]
    # Instantiate lasso_regression_params.
    lasso_regression_params_inst = lasso_regression_params()
    # Instantiate lasso_regression.
    lasso_regression_inst = lasso_regression(lasso_regression_params_inst.model)
    # Fit model with train data.
    lasso_regression_inst.fit(X_train, y_train)
    # Test the fitted model with test data.
    y_pred = lasso_regression_inst.predict(X_test)
    accuracy = np.sum(y_pred == y_test) / y_test.shape[0]
    print("The accuracy of predicted y is {:.3f}.".format(accuracy))

