import numpy as np

from ml_deeco.estimators import Estimator


class NoEstimator(Estimator):
    """
    An estimator mainly for debugging purposes (as estimate cannot be created without an estimator).
    It does not produce any training logs (on standard output) or outputs (does not save training data).
    Predicts 0 for each target.
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
