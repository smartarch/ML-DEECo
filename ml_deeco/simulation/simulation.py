import abc
from datetime import datetime
from typing import List, Tuple, TYPE_CHECKING

from ml_deeco.simulation import Configuration
from ml_deeco.utils import verbosePrint

if TYPE_CHECKING:
    from ml_deeco.simulation import Ensemble, Component
    from ml_deeco.estimators import Estimator


def materialize_ensembles(components, ensembles):
    """
    Performs the materialization of all ensembles. That includes actuating the materialized ensembles and collecting
    data for the estimates.

    Parameters
    ----------
    components : List['Component']
        All components in the system.
    ensembles : List['Ensemble']
        All potential ensembles in the system.

    Returns
    -------
    List['Ensemble']
        The materialized ensembles.
    """
    materializedEnsembles = []

    potentialEnsembles = sorted(ensembles)
    for ens in potentialEnsembles:
        if ens.materialize(components, materializedEnsembles):
            materializedEnsembles.append(ens)
            ens.actuate()
    for ens in potentialEnsembles:
        ens.collectEstimatesData(components)

    return materializedEnsembles


def actuate_components(components):
    """
    Performs component actuation. Runs the actuate function on all components and collects the data for the estimates.

    Parameters
    ----------
    components : List['Component']
        All components in the system.
    """
    for component in components:
        component.actuate()

    for component in components:
        component.collectEstimatesData()


class Experiment(abc.ABC):

    def __init__(self, config: Configuration = None):

        self.currentTimeStep = 0

        if config is not None:
            self.config = config
        else:
            self.config = Configuration()

        self.useBaselines = True
        self.baselineEstimator = 0
        self.estimators: List['Estimator'] = []
        self.createEstimators()

    # helper functions

    def createEstimators(self):
        if 'estimators' in self.config.__dict__:
            for estimatorName, estimator in self.config.estimators.items():
                className = estimator['class']
                constructorArgs = estimator['args'] if 'args' in estimator else {}
                components = className.split('.')
                module = __import__('.'.join(components[:-1]))
                for component in components[1:]:
                    module = getattr(module, component)
                classCreator = module
                if classCreator is None:
                    raise RuntimeError(f"Class {className} not found.")

                constructorArgs = {
                    **constructorArgs,
                    'baseFolder': self.config.output,
                    'experiment': self,
                }

                obj = classCreator(**constructorArgs)

                if 'baseline' in estimator:
                    obj.baseline = estimator['baseline']

                self.__dict__[estimatorName] = obj

    def appendEstimator(self, estimator):
        self.estimators.append(estimator)

    def initEstimators(self):
        """Initialize the estimators. This has to be called after the components and ensembles are imported and before
        the simulation is run."""
        for est in self.estimators:
            est.init()

    # user-supplied callbacks

    @abc.abstractmethod
    def prepareSimulation(self, iteration: int, run: int) -> Tuple[List['Component'], List['Ensemble']]:
        """
        Prepares the components and ensembles for the simulation run.

        Parameters
        ----------
        iteration
            Number of the current iteration.
        run
            Number of the current simulation run (in the current iteration).

        Returns
        -------
        (components, ensembles)
            List of components, list of potential ensembles.
        """
        return [], []

    def prepareIteration(self, iteration: int):
        """
        Performed at the beginning of each iteration.

        Parameters
        ----------
        iteration
            Number of the current iteration.
        """
        pass

    def iterationCallback(self, iteration: int):
        """
        Performed at the end of each iteration (after the training of Estimators).

        Parameters
        ----------
        iteration
            Number of the current iteration.
        """
        pass

    def simulationCallback(self, components: List['Component'], ensembles: List['Ensemble'], iteration: int, run: int):
        """
        Performed after each simulation run.

        Parameters
        ----------
        components
            List of components (returned by `prepareSimulation`).
        ensembles
            List of potential ensembles (returned by `prepareSimulation`).
        iteration
            Number of the current iteration.
        run
            Number of the current simulation (in the current iteration).
        """
        pass

    def stepCallback(self, components: List['Component'], materialized_ensembles: List['Ensemble'], step: int):
        """
        Performed after each simulation.

        Parameters
        ----------
        components
            List of all components in the system.
        materialized_ensembles
            List of materialized ensembles (in this time step).
        step
            Current time step.
        """
        pass

    # running the simulation and experiment

    def runSimulation(self, components: List['Component'], ensembles: List['Ensemble']):
        """
        Performs one runs of the simulation with `components` and `ensembles` for `self.config.steps` steps.

        Parameters
        ----------
        components
            All components in the system.
        ensembles
            All potential ensembles in the system.
        """

        for step in range(self.config.steps):

            verbosePrint(f"Step {step + 1}:", 3)
            self.currentTimeStep = step

            materializedEnsembles = materialize_ensembles(components, ensembles)
            actuate_components(components)

            self.stepCallback(components, materializedEnsembles, step)

    def run(self):
        """
        Runs `self.config.iterations` iterations of the experiment.
        Each iteration consist of running the simulation `self.config.simulations` times (each simulation is run for
        `self.config.steps` steps) and then performing training of the Estimators (ML models)."""

        self.initEstimators()

        for iteration in range(self.config.iterations):
            verbosePrint(f"Iteration {iteration + 1} started at {datetime.now()}:", 1)
            self.prepareIteration(iteration)

            for simulation in range(self.config.simulations):
                verbosePrint(f"Simulation {simulation + 1} started at {datetime.now()}:", 2)

                components, ensembles = self.prepareSimulation(iteration, simulation)

                self.runSimulation(components, ensembles)

                self.simulationCallback(components, ensembles, iteration, simulation)

            for estimator in self.estimators:
                estimator.endIteration()

            self.iterationCallback(iteration)

            self.useBaselines = False
