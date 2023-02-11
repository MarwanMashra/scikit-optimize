"""Machine learning extensions for model-based optimization."""

from .forest import RandomForestRegressor
from .forest import ExtraTreesRegressor
from .gaussian_process import GaussianProcessRegressor
from .gbrt import GradientBoostingQuantileRegressor
from .xgboost import XGBRegressor


__all__ = (
    "RandomForestRegressor",
    "ExtraTreesRegressor",
    "GradientBoostingQuantileRegressor",
    "GaussianProcessRegressor",
    "XGBRegressor",
)
