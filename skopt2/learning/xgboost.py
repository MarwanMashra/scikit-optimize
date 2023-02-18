import numpy as np

from xgboost import XGBRegressor
from sklearn.base import BaseEstimator, RegressorMixin
from functools import partial
from sklearn.utils import check_random_state
from sklearn.base import clone
from joblib import Parallel, delayed


def _parallel_fit(regressor, X, y):
    return regressor.fit(X, y)


class XGBRegressor(BaseEstimator, RegressorMixin):
    """Predict several quantiles with one estimator.

    This is a wrapper around `GradientBoostingRegressor`'s quantile
    regression that allows you to predict several `quantiles` in
    one go.

    Parameters
    ----------
    quantiles : array-like
        Quantiles to predict. By default the 16, 50 and 84%
        quantiles are predicted.

    base_estimator : GradientBoostingRegressor instance or None (default)
        Quantile regressor used to make predictions. Only instances
        of `GradientBoostingRegressor` are supported. Use this to change
        the hyper-parameters of the estimator.

    n_jobs : int, default=1
        The number of jobs to run in parallel for `fit`.
        If -1, then the number of jobs is set to the number of cores.

    random_state : int, RandomState instance, or None (default)
        Set random state to something other than None for reproducible
        results.
    """

    def __init__(
        self,
        quantiles=[0.16, 0.5, 0.84],
        base_estimator=None,
        n_jobs=1,
        random_state=None,
    ):
        self.quantiles = quantiles
        self.random_state = random_state
        self.base_estimator = base_estimator
        self.n_jobs = n_jobs

    def fit(self, X, y):

        rng = check_random_state(self.random_state)

        if self.base_estimator is None:
            base_estimator = XGBRegressor()
        else:
            base_estimator = self.base_estimator

        base_estimator.set_params(random_state=rng)

        regressors = []
        for q in self.quantiles:
            regressor = clone(base_estimator)
            regressor.set_params(
                objective=partial(
                    XGBRegressor.quantile_loss,
                    alpha=q,
                    delta=self.quant_delta,
                    threshold=self.quant_thres,
                    var=self.quant_var,
                )
            )
            regressors.append(regressor)

        self.regressors_ = Parallel(n_jobs=self.n_jobs, backend="threading")(
            delayed(_parallel_fit)(regressor, X, y) for regressor in regressors
        )

        return self

    def predict(self, X, return_std=False, return_quantiles=False):
        """Predict.

        Predict `X` at every quantile if `return_std` is set to False.
        If `return_std` is set to True, then return the mean
        and the predicted standard deviation, which is approximated as
        the (0.84th quantile - 0.16th quantile) divided by 2.0

        Parameters
        ----------
        X : array-like, shape=(n_samples, n_features)
            where `n_samples` is the number of samples
            and `n_features` is the number of features.
        """
        predicted_quantiles = np.asarray([rgr.predict(X) for rgr in self.regressors_])
        if return_quantiles:
            return predicted_quantiles.T

        elif return_std:
            std_quantiles = [0.16, 0.5, 0.84]
            is_present_mask = np.in1d(std_quantiles, self.quantiles)
            if not np.all(is_present_mask):
                raise ValueError(
                    "return_std works only if the quantiles during "
                    "instantiation include 0.16, 0.5 and 0.84"
                )
            low = self.regressors_[self.quantiles.index(0.16)].predict(X)
            high = self.regressors_[self.quantiles.index(0.84)].predict(X)
            mean = self.regressors_[self.quantiles.index(0.5)].predict(X)
            return mean, ((high - low) / 2.0)

        # return the mean
        return self.regressors_[self.quantiles.index(0.5)].predict(X)

    @staticmethod
    def quantile_loss(y_true, y_pred, alpha, delta, threshold, var):
        x = y_true - y_pred
        grad = (
            (x < (alpha - 1.0) * delta) * (1.0 - alpha)
            - ((x >= (alpha - 1.0) * delta) & (x < alpha * delta)) * x / delta
            - alpha * (x > alpha * delta)
        )
        hess = ((x >= (alpha - 1.0) * delta) & (x < alpha * delta)) / delta

        grad = (np.abs(x) < threshold) * grad - (np.abs(x) >= threshold) * (
            2 * np.random.randint(2, size=len(y_true)) - 1.0
        ) * var
        hess = (np.abs(x) < threshold) * hess + (np.abs(x) >= threshold)
        return grad, hess

    # @staticmethod
    # def original_quantile_loss(y_true, y_pred, alpha, delta):
    #     x = y_true - y_pred
    #     grad = (
    #         (x < (alpha - 1.0) * delta) * (1.0 - alpha)
    #         - ((x >= (alpha - 1.0) * delta) & (x < alpha * delta)) * x / delta
    #         - alpha * (x > alpha * delta)
    #     )
    #     hess = ((x >= (alpha - 1.0) * delta) & (x < alpha * delta)) / delta
    #     return grad, hess

    # def score(self, X, y):
    #     y_pred = super().predict(X)
    #     score = XGBRegressor.quantile_score(y, y_pred, self.quant_alpha)
    #     score = 1.0 / score
    #     return score

    # @staticmethod
    # def quantile_score(y_true, y_pred, alpha):
    #     score = XGBRegressor.quantile_cost(x=y_true - y_pred, alpha=alpha)
    #     score = np.sum(score)
    #     return score

    # @staticmethod
    # def quantile_cost(x, alpha):
    #     return (alpha - 1.0) * x * (x < 0) + alpha * x * (x >= 0)

    # @staticmethod
    # def get_split_gain(gradient, hessian, l=1):
    #     split_gain = list()
    #     for i in range(gradient.shape[0]):
    #         split_gain.append(
    #             np.sum(gradient[:i]) / (np.sum(hessian[:i]) + l)
    #             + np.sum(gradient[i:]) / (np.sum(hessian[i:]) + l)
    #             - np.sum(gradient) / (np.sum(hessian) + l)
    #         )

    #     return np.array(split_gain)
