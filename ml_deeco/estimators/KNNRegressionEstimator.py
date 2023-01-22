from sklearn import neighbors

from ml_deeco.estimators.ScikitEstimator import ScikitEstimator


class KNNRegressionEstimator(ScikitEstimator):
    """
    Estimator using sklearn.neighbors.KNeighborsRegressor
    (https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsRegressor.html)
    """

    @property
    def estimatorName(self):
        return f"KNN Regression"

    def __init__(self, experiment, k=3, params=None, **kwargs):
        super().__init__(experiment, **kwargs)
        if params is None:
            params = {}
        self._model = neighbors.KNeighborsRegressor(n_neighbors=k, **params)
