from datetime import datetime
from typing import Optional, Callable, List, Tuple, TYPE_CHECKING

from ml_deeco.utils import verbosePrint

if TYPE_CHECKING:
    from ml_deeco.simulation import Ensemble, Component


class SimulationGlobals:
    """
    Storage for the global simulation data such as a list of all estimators and the current time step.

    This simplifies the implementation as it allows for instance the TimeEstimate to access the current time step of the simulation.
    """

    def __init__(self):
        if 'SIMULATION_GLOBALS' in locals():
            raise RuntimeError("Do not create a new instance of the SimulationGlobals. Use the SIMULATION_GLOBALS global variable instead.")
        self.estimators = []
        self.currentTimeStep = 0
        self.useBaselines = True  # TODO: documentation

    def initEstimators(self):
        """Initialize the estimators. This has to be called after the components and ensembles are imported and before the simulation is run."""
        for est in self.estimators:
            est.init()


SIMULATION_GLOBALS = SimulationGlobals()


def materialize_ensembles(components, ensembles):
    """
    Performs the materialization of all ensembles. That includes actuating the materialized ensembles and collecting data for the estimates.

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
        verbosePrint(f"{component}", 4)
    for component in components:
        component.collectEstimatesData()


def run_simulation(
    components: List['Component'],
    ensembles: List['Ensemble'],
    steps: int,
    stepCallback: Optional[Callable[[List['Component'], List['Ensemble'], int], None]] = None
):
    """
    Runs the simulation with `components` and `ensembles` for `steps` steps.

    Parameters
    ----------
    components
        All components in the system.
    ensembles
        All potential ensembles in the system.
    steps
        Number of steps to run.
    stepCallback
        This function is called after each simulation step. It can be used for example to log data from the simulation. The parameters are:
            - list of all components in the system,
            - list of materialized ensembles (in this time step),
            - current time step (int).
    """

    for step in range(steps):

        verbosePrint(f"Step {step + 1}:", 3)
        SIMULATION_GLOBALS.currentTimeStep = step

        materializedEnsembles = materialize_ensembles(components, ensembles)
        actuate_components(components)

        if stepCallback:
            stepCallback(components, materializedEnsembles, step)


def run_experiment(
    iterations: int,
    simulations: int,
    steps: int,
    prepareSimulation: Callable[[int, int], Tuple[List['Component'], List['Ensemble']]],
    prepareIteration: Optional[Callable[[int], None]] = None,
    iterationCallback: Optional[Callable[[int], None]] = None,
    simulationCallback: Optional[Callable[[List['Component'], List['Ensemble'], int, int], None]] = None,
    stepCallback: Optional[Callable[[List['Component'], List['Ensemble'], int], None]] = None
):
    """
    Runs `iterations` iteration of the experiment. Each iteration consist of running the simulation `simulations` times (each simulation is run for `steps` steps) and then performing training of the Estimator (ML model).

    Parameters
    ----------
    iterations
        Number of iterations to run.
    simulations
        Number of simulations to run in each iteration.
    steps
        Number of steps to perform in each simulation.
    prepareSimulation
        Prepares the components and ensembles for the simulation.
        Parameters:
            - current iteration,
            - current simulation (in the current iteration).
        Returns:
            - list of components,
            - list of potential ensembles.
    prepareIteration
        Performed at the beginning of each iteration.
        Parameters:
            - current iteration.
    iterationCallback
        Performed at the end of each iteration (after the training of Estimators).
        Parameters:
            - current iteration.
    simulationCallback
        Performed after each simulation.
        Parameters:
            - list of components (returned by `prepareSimulation`),
            - list of potential ensembles (returned by `prepareSimulation`),
            - current iteration,
            - current simulation (in the current iteration).
    stepCallback
        This function is called after each simulation step. It can be used for example to log data from the simulation. The parameters are:
            - list of all components in the system,
            - list of materialized ensembles (in this time step),
            - current time step (int).
    """

    SIMULATION_GLOBALS.initEstimators()

    for iteration in range(iterations):
        verbosePrint(f"Iteration {iteration + 1} started at {datetime.now()}:", 1)
        if prepareIteration:
            prepareIteration(iteration)

        for simulation in range(simulations):
            verbosePrint(f"Simulation {simulation + 1} started at {datetime.now()}:", 2)

            components, ensembles = prepareSimulation(iteration, simulation)

            run_simulation(components, ensembles, steps, stepCallback)

            if simulationCallback:
                simulationCallback(components, ensembles, iteration, simulation)

        for estimator in SIMULATION_GLOBALS.estimators:
            estimator.endIteration()

        if iterationCallback:
            iterationCallback(iteration)

        SIMULATION_GLOBALS.useBaselines = False
