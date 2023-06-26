import os
# local dep
if __name__ == "__main__":
    import os, sys
    sys.path.insert(0, os.pardir)
from utils import DotDict

__all__ = [
    "logistic_regression_params",
]

class logistic_regression_params(DotDict):
    """
    This contains one single object that generates a dictionary of parameters,
    which is provided to `logistic_regression` on initailization.
    """
    # Internal macro parameter.
    _precision = "float32"

    def __init__(self, dataset="meg_liu2019cell"):
        """
        Initialize `logistic_regression_params` object.
        """
        ## First call super class init function to set up `DotDict`
        ## style object and inherit it's functionality.
        super(logistic_regression_params, self).__init__()

        ## Generate all parameters hierarchically.
        # -- Model parameters
        self.model = logistic_regression_params._gen_model_params()
        # -- Train parameters
        self.train = logistic_regression_params._gen_train_params(dataset)

    """
    generate funcs
    """
    ## def _gen_model_* funcs
    # def _gen_model_params func
    @staticmethod
    def _gen_model_params():
        """
        Generate model parameters.
        """
        # Initialize model_params.
        model_params = DotDict()

        ## -- Normal parameters
        # Specify the norm of the penalty:
        # 1) None: no penalty is added;
        # 2) 'l2': add a L2 penalty term and it is the default choice;
        # 3) 'l1': add a L1 penalty term;
        # 4) 'elasticnet': both L1 and L2 penalty terms are added.
        # Note:  Some penalties may not work with some solvers. See the parameter solver below, to
        # know the compatibility between the penalty and solver.
        model_params.penalty = "elasticnet"
        # Dual or primal formulation. Dual formulation is only implemented for l2 penalty with liblinear
        # solver. Prefer `dual` = `False` when `n_samples` > `n_features`.
        model_params.dual = False
        # The tolerance for the optimization: if the updates are smaller than `tol`, the optimization code
        # checks the dual gap for optimality and continues until it is smaller than `tol`, see Notes below.
        model_params.tol = 1e-4
        # Inverse of regularization strength; must be a positive float. Like in support vector machines,
        # smaller values specify stronger regularization. And `1 / (2 * C)` equals to `alpha`.
        model_params.C = 500.
        # Whether to calculate the intercept for this model. If set to `False`, no intercept will be
        # used in calculations (i.e. data is expected to be centered).
        model_params.fit_intercept = True
        # Useful only when the solver 'liblinear' is used and `self.fit_intercept` is set to True. In this case,
        # x becomes [x, self.intercept_scaling], i.e. a "synthetic" feature with constant value equal to `intercept_scaling`
        # is appended to the instance vector. The intercept becomes `intercept_scaling` * `synthetic_feature_weight`.
        # Note: The synthetic feature weight is subject to l1/l2 regularization as all other features. To lessen
        # the effect of regularization on synthetic feature weight (and therefore on the intercept)
        # `intercept_scaling` has to be increased.
        model_params.intercept_scaling = 1
        # Weights associated with classes in the form {class_label: weight}. If not given, all classes are supposed
        # to have weight one. The "balanced" mode uses the values of y to automatically adjust weights inversely
        # proportional to class frequencies in the input data as `n_samples` / (`n_classes` * `np.bincount(y)`).
        # Note that these weights will be multiplied with `sample_weight` (passed through the fit method)
        # if `sample_weight` is specified.
        model_params.class_weight = None
        # Algorithm to use in the optimization problem. Default is 'lbfgs'. To choose a solver, you might
        # want to consider the following aspects:
        # 1) For small datasets, 'liblinear' is a good choice, whereas 'sag' and 'saga' are faster for large ones;
        # 2) For multiclass problems, only 'newton-cg', 'sag', 'saga' and 'lbfgs' handle multinomial loss;
        # 3) 'liblinear' and is limited to one-versus-rest schemes.
        # 4) 'newton-cholesky' is a good choice for `n_samples` >> `n_features`, especially with one-hot encoded
        #    categorical features with rare categories. Note that it is limited to binary classification and the
        #    one-versus-rest reduction for multiclass classification. Be aware that the memory usage of this solver
        #    has a quadratic dependency on `n_features` because it explicitly computes the Hessian matrix.
        model_params.solver = "saga"
        # Used when `solver` == 'sag' or 'saga' to shuffle the data.
        model_params.random_state = None
        # Maximum number of iterations taken for the solvers to converge.
        model_params.max_iter = int(1e2)
        # If the option chosen is 'ovr', then a binary problem is fit for each label. For 'multinomial' the loss
        # minimised is the multinomial loss fit across the entire probability distribution, even when the data is
        # binary. 'multinomial' is unavailable when `solver` = 'liblinear'. 'auto' selects 'ovr' if the data is binary,
        # or if `solver` = 'liblinear', and otherwise selects 'multinomial'.
        model_params.multi_class = "auto"
        # For the liblinear and lbfgs solvers set verbose to any positive number for verbosity.
        model_params.verbose = 0
        # When set to `True`, reuse the solution of the previous call to fit as initialization, otherwise,
        # just erase the previous solution. Useless for liblinear solver.
        model_params.warm_start = False
        # Number of CPU cores used when parallelizing over classes if `multi_class` = 'ovr'. This parameter is
        # ignored when the solver is set to 'liblinear' regardless of whether 'multi_class' is specified or not.
        # None means 1 unless in a `joblib.parallel_backend` context. -1 means using all processors.
        model_params.n_jobs = None
        # The Elastic-Net mixing parameter, with 0 <= `l1_ratio` <= 1. Only used if `penalty` = 'elasticnet'.
        # Setting `l1_ratio` = 0 is equivalent to using `penalty` = 'l2', while setting `l1_ratio` = 1 is
        # equivalent to using `penalty` = 'l1'. For 0 < `l1_ratio` <1, the penalty is a combination of L1 and L2.
        model_params.l1_ratio = 0.5

        # Return the final `model_params`.
        return model_params

    ## def _gen_train_* funcs
    # def _gen_train_params func
    @staticmethod
    def _gen_train_params(dataset):
        """
        Generate train parameters.
        """
        # Initialize train parameters.
        train_params = DotDict()

        ## -- Normal parameters
        # The type of dataset.
        train_params.dataset = dataset
        # The ratio of train dataset. The rest is test dataset.
        train_params.train_ratio = 0.8

        # Return the final `train_params`.
        return train_params

if __name__ == "__main__":
    # Instantiate logistic_regression_params.
    logistic_regression_params_inst = logistic_regression_params(dataset="meg_liu2019cell")

