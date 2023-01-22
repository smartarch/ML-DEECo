from .features import *
from .estimate import *
from .estimator import *

# ML models
from .ConstantEstimator import ConstantEstimator
from .NoEstimator import NoEstimator

try:
    from .NeuralNetworkEstimator import NeuralNetworkEstimator
except ImportError:  # tf not installed
    class NeuralNetworkEstimator:
        def __init__(self, *args, **kwargs):
            raise ImportError("tensorflow not installed")
try:
    from .LinearRegressionEstimator import LinearRegressionEstimator
    from .KNNRegressionEstimator import KNNRegressionEstimator
    from .GaussianProcessEstimator import GaussianProcessEstimator
    from .SupportVectorRegressionEstimator import SupportVectorRegressionEstimator
except ImportError:  # sklearn not installed
    class MissingScikitEstimator:
        def __init__(self, *args, **kwargs):
            raise ImportError("sklearn not installed")

    class LinearRegressionEstimator(MissingScikitEstimator):
        pass

    class KNNRegressionEstimator(MissingScikitEstimator):
        pass

    class GaussianProcessEstimator(MissingScikitEstimator):
        pass

    class SupportVectorRegressionEstimator(MissingScikitEstimator):
        pass
