from sklearn import svm

from ml_deeco.estimators.ScikitEstimator import ScikitEstimator


class SupportVectorRegressionEstimator(ScikitEstimator):
    """
    Estimator using Epsilon-Support Vector Regression: sklearn.svm.SVR
    (https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVR.html)
    """

    @property
    def estimatorName(self):
        return f"Support Vector Regression"

    def __init__(self, experiment, params=None, **kwargs):
        super().__init__(experiment, **kwargs)
        if params is None:
            params = {}
        self._model = svm.SVR(**params)
