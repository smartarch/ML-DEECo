import numpy as np

from ml_deeco.estimators import Estimator


class ConstantEstimator(Estimator):
    """
    Predicts a given constant for each target.
    """

    def __init__(self, value=0., **kwargs):
        super().__init__(**kwargs)
        self._value = value

    @property
    def estimatorName(self):
        return f"ConstantEstimator({self._value})"

    def predict(self, x):
        numTargets = sum((feature.getNumFeatures() for _, feature, _ in self._targets))
        return np.full([numTargets], self._value)
