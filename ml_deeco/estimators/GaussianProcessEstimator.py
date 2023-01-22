from sklearn import gaussian_process

from ml_deeco.estimators.ScikitEstimator import ScikitEstimator


class GaussianProcessEstimator(ScikitEstimator):
    """
    Estimator using sklearn.gaussian_process.GaussianProcessRegressor
    (https://scikit-learn.org/stable/modules/generated/sklearn.gaussian_process.GaussianProcessRegressor.html)
    """

    @property
    def estimatorName(self):
        return f"Gaussian Process"

    def __init__(self, experiment, params=None, **kwargs):
        super().__init__(experiment, **kwargs)
        if params is None:
            params = {}
        self._model = gaussian_process.GaussianProcessRegressor(**params)
