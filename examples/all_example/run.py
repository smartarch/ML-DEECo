# This is to disable logging and use of GPU in TensorFLow
import os
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")  # Report only TF errors by default
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # Disable GPU in TF. The models are small, so it is actually faster to use the CPU.
import tensorflow as tf
# Fix random seed
tf.random.set_seed(42)

from truck import Truck
from package_ensemble import PackageEnsemble

from ml_deeco.simulation import Point, run_experiment
from ml_deeco.utils import setVerboseLevel

setVerboseLevel(2)


truck = Truck(Point(0, 0))
packageEnsemble = PackageEnsemble(Point(9, 0))


def prepareSimulation(iteration, simulation):
    """This is called before each simulation."""

    # we initialize the truck
    global truck
    truck = Truck(Point(0, 0))

    # and return the lists of components (only the truck) and ensembles (one instance of PackageEnsemble)
    components = [truck]
    ensembles = [packageEnsemble]
    return components, ensembles


run_experiment(
    iterations=2,   # we run two iterations -- the ML model trains between iterations
    simulations=1,  # one simulation in each iteration
    steps=80,       # the simulation is run for 80 steps
    prepareSimulation=prepareSimulation,
)
