from sklearn import linear_model

from ml_deeco.estimators.ScikitEstimator import ScikitEstimator


class LinearRegressionEstimator(ScikitEstimator):
    """
    Estimator using sklearn.linear_model.LinearRegression
    (https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html)
    """

    @property
    def estimatorName(self):
        return f"Linear Regression"

    def __init__(self, experiment, **kwargs):
        super().__init__(experiment, **kwargs)
        self._model = linear_model.LinearRegression()
