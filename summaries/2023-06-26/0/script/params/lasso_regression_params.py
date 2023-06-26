# local dep
if __name__ == "__main__":
    import os, sys
    sys.path.insert(0, os.pardir)
from utils import DotDict

__all__ = [
    "lasso_regression_params",
]

class lasso_regression_params(DotDict):
    """
    This contains one single object that generates a dictionary of parameters,
    which is provided to `lasso_regression` on initailization.
    """
    # Internal macro parameter.
    _precision = "float32"

    def __init__(self):
        """
        Initialize `lasso_regression_params` object.
        """
        ## First call super class init function to set up `DotDict`
        ## style object and inherit it's functionality.
        super(lasso_regression_params, self).__init__()

        ## Generate all parameters hierarchically.
        # -- Model parameters
        self.model = lasso_regression_params._gen_model_params()

    """
    generate funcs
    """
    ## def _gen_model_* func
    # def _gen_model_params func
    @staticmethod
    def _gen_model_params():
        """
        Generate model parameters.
        """
        # Initialize model_params.
        model_params = DotDict()

        ## -- Normal parameters
        # Constant that multiplies the L1 term, controlling regularization strength. `alpha` must be a non-negative
        # float i.e. in [0, inf). When `alpha` = 0, the objective is equivalent to ordinary least squares, solved
        # by the `LinearRegression` object. For numerical reasons, using `alpha` = 0 with the Lasso object is not
        # advised. Instead, you should use the `LinearRegression` object. `alpha` corresponds to `1 / (2C)` in
        # other linear models such as `LogisticRegression` or `LinearSVC`.
        model_params.alpha = 0.001
        # Whether to calculate the intercept for this model. If set to `False`, no intercept will be
        # used in calculations (i.e. data is expected to be centered).
        model_params.fit_intercept = True
        # Whether to use a precomputed Gram matrix to speed up calculations. The Gram matrix can also
        # be passed as argument. For sparse input this option is always `False` to preserve sparsity.
        model_params.precompute = False
        # If `True`, X will be copied; else, it may be overwritten.
        model_params.copy_X = True
        # The maximum number of iterations.
        model_params.max_iter = int(1e3)
        # The tolerance for the optimization: if the updates are smaller than `tol`, the optimization code
        # checks the dual gap for optimality and continues until it is smaller than `tol`, see Notes below.
        model_params.tol = 1e-4
        # When set to `True`, reuse the solution of the previous call to fit as initialization,
        # otherwise, just erase the previous solution.
        model_params.warm_start = False
        # When set to `True`, forces the coefficients to be positive.
        model_params.positive = False
        # The seed of the pseudo random number generator that selects a random feature to update. Used when
        # `selection` == 'random'. Pass an int for reproducible output across multiple function calls.
        model_params.random_state = None
        # If set to 'random', a random coefficient is updated every iteration rather than looping over
        # features sequentially by default. This (setting to 'random') often leads to significantly
        # faster convergence especially when `tol` is higher than 1e-4.
        model_params.selection = "cyclic"

        # Return the final `model_params`.
        return model_params

if __name__ == "__main__":
    # Instantiate lasso_regression_params.
    lasso_regression_params_inst = lasso_regression_params()

