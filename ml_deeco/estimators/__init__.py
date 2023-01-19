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
except ImportError:  # sklearn not installed
    class LinearRegressionEstimator:
        def __init__(self, *args, **kwargs):
            raise ImportError("sklearn not installed")
