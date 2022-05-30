from sklearn import linear_model

from ml_deeco.estimators import Estimator


class LinearRegressionEstimator(Estimator):
    """
    use a simple ml method to perform regression
    """
    @property
    def estimatorName(self):
        return f"Linear Regression"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._model = linear_model.LinearRegression()

    def train(self, x, y):
        self._model.fit(x, y)

    def predict(self, x):
        return self._model.predict([x])[0]
