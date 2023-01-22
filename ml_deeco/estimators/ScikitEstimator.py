from abc import ABC

from ml_deeco.estimators import Estimator


class ScikitEstimator(Estimator, ABC):
    """
    Base class for Scikit-Learn-based estimators
    """

    def __init__(self, experiment, **kwargs):
        super().__init__(experiment, **kwargs)
        self._model = ...

    def train(self, X, y):
        self._model.fit(X, y)

    def predict(self, x):
        return self._model.predict([x])[0]

    def predictBatch(self, X):
        return self._model.predict(X).reshape(-1, 1)

    def saveModel(self, suffix=""):
        from joblib import dump
        suffix = str(suffix)
        if suffix:
            filename = f"model_{suffix}.pkl"
        else:
            filename = "model.pkl"
        dump(self._model, self._outputFolder / filename)

    def loadModel(self, modelPath=None):
        from joblib import load
        if modelPath is None:
            modelPath = self._outputFolder / "model.pkl"
        self._model = load(modelPath)
