from ml_deeco.estimators import LinearRegressionEstimator
from plot import drawPlot

from ml_deeco.simulation import Point2D, Experiment, Configuration
from ml_deeco.utils import setVerboseLevel, Log

setVerboseLevel(2)


class TruckExperiment(Experiment):

    log: Log = None  # type: ignore

    def __init__(self, config):
        super().__init__(config)
        self.truck = None

    def prepareSimulation(self, iteration, simulation):
        """This is called before each simulation."""

        # create the truck component
        from truck import Truck

        self.truck = Truck(Point2D(0, 0))
        if iteration > 0:
            self.truck.useEstimate = True

        # prepare a log
        self.log = Log({
            "location": ".0f",
            "battery": ".2f",
            "future_battery": ".2f",
            "state": None
        })

        # initialize the ensembles
        from package_ensemble import PackageEnsemble
        packageEnsemble = PackageEnsemble(Point2D(9, 0))

        # and return the lists of components (only the truck) and ensembles (one instance of PackageEnsemble)
        components = [self.truck]
        ensembles = [packageEnsemble]
        return components, ensembles

    def stepCallback(self, components, materializedEnsembles, step):
        """We perform logging of the simulation steps to later draw charts."""
        self.log.register([
            self.truck.location.x,
            self.truck.fuel,
            self.truck.fuelEstimate(),
            self.truck.state
        ])

    def simulationCallback(self, components, ensembles, iteration, simulation):
        """This is called when the simulation completes."""
        # we save the logged data and draw a plot for the first simulation in each iteration
        if simulation == 0:
            self.log.export(f"results/log_{iteration + 1}.csv")
            drawPlot(iteration + 1, self.log)


if __name__ == '__main__':

    experiment = TruckExperiment(
        config=Configuration(
            iterations=2,   # we run two iterations -- the ML model trains between iterations
            simulations=1,  # one simulation in each iteration
            steps=80,       # the simulation is run for 80 steps
        )
    )

    # prepare the neural network
    experiment.truckFuelEstimator = LinearRegressionEstimator(
        experiment,
        outputFolder="results/fuel", name="Truck Fuel"
    )

    # import the Truck component to initialize the estimate
    from truck import Truck
    Truck.initEstimates(experiment)

    experiment.run()
