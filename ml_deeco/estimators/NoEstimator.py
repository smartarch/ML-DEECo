import numpy as np

from ml_deeco.estimators import Estimator


class NoEstimator(Estimator):
    """
    Does not produce any training logs or outputs. Predicts 0 for each target.
    """

    def __init__(self, experiment, **kwargs):
        super().__init__(experiment, **kwargs, outputFolder=None, skipEndIteration=True, printLogs=False)

    def train(self, X, Y):
        pass

    @property
    def estimatorName(self):
        return f"NoEstimator"

    def predict(self, x):
        numTargets = sum((feature.getNumFeatures() for _, feature, _ in self._targets))
        return np.zeros([numTargets])
