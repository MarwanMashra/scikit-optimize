import numpy as np
from xgboost import XGBRegressor as _sk_XGBRegressor


def _return_std(X, trees, predictions, min_variance):
    # This derives std(y | x) as described in 4.3.2 of arXiv:1211.0906
    std = np.zeros(len(X))

    for tree in trees:
        var_tree = tree.tree_.impurity[tree.apply(X)]

        # This rounding off is done in accordance with the
        # adjustment done in section 4.3.3
        # of http://arxiv.org/pdf/1211.0906v2.pdf to account
        # for cases such as leaves with 1 sample in which there
        # is zero variance.
        var_tree[var_tree < min_variance] = min_variance
        mean_tree = tree.predict(X)
        std += var_tree + mean_tree**2

    std /= len(trees)
    std -= predictions**2.0
    std[std < 0.0] = 0.0
    std = std**0.5
    return std


class XGBRegressor(_sk_XGBRegressor):
    def __init__(
        self,
        max_depth=3,
        learning_rate=0.1,
        n_estimators=300,
        silent=True,
        objective="binary:logistic",
        booster="gbtree",
        n_jobs=-1,
        nthread=None,
        gamma=0,
        min_child_weight=1,
        max_delta_step=0,
        subsample=1,
        colsample_bytree=1,
        colsample_bylevel=1,
        reg_alpha=0,
        reg_lambda=1,
        scale_pos_weight=1,
        base_score=0.5,
        random_state=0,
        seed=None,
        missing=None,
        min_variance=0.0,
    ):
        self.min_variance = min_variance
        super(XGBRegressor, self).__init__(
            max_depth=max_depth,
            learning_rate=learning_rate,
            n_estimators=n_estimators,
            silent=silent,
            objective=objective,
            booster=booster,
            n_jobs=n_jobs,
            nthread=nthread,
            gamma=gamma,
            min_child_weight=min_child_weight,
            max_delta_step=max_delta_step,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            colsample_bylevel=colsample_bylevel,
            reg_alpha=reg_alpha,
            reg_lambda=reg_lambda,
            scale_pos_weight=scale_pos_weight,
            base_score=base_score,
            random_state=random_state,
            seed=seed,
            missing=missing,
            min_variance=min_variance,
        )

    def predict(self, X, return_std=False):
        mean = super(XGBRegressor, self).predict(X)

        if return_std:
            std = _return_std(X, self.estimators_, mean, self.min_variance)
            return mean, std

        return mean
