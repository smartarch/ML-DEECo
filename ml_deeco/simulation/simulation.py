from datetime import datetime
from typing import Optional, Callable, List, Tuple, TYPE_CHECKING

from ml_deeco.utils import verbosePrint, readYaml, Validators
if TYPE_CHECKING:
    from ml_deeco.simulation import Ensemble, Component

from pathlib import Path

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

class Simulation:
    def __init__ (self,        
        prepareSimulation: Callable[[int, int], Tuple[List['Component'], List['Ensemble']]],
        prepareIteration: Optional[Callable[[int], None]] = None,
        iterationCallback: Optional[Callable[[int], None]] = None,
        simulationCallback: Optional[Callable[[List['Component'], List['Ensemble'], int, int], None]] = None,
        stepCallback: Optional[Callable[[List['Component'], List['Ensemble'], int], None]] = None,
        iterations: int = 0,
        simulations: int = 0,
        configFile: str = None,
        baseFolder: Path = None ):

        self.prepareSimulation = prepareSimulation
        self.prepareIteration = prepareIteration
        self.iterationCallback = iterationCallback
        self.simulationCallback = simulationCallback
        self.stepCallback = stepCallback
        
        self.iterations = iterations
        self.simulations = simulations
        self.estimators = {}
        self.baseFolder = baseFolder 

        if configFile is not None:
            self.processConfigFile(configFile)


    def processConfigFile(self, configFile):
        configDict = readYaml(configFile)

        for intArgument in ['iterations','simulations']:
            if intArgument in configDict:
                Validators.validateType(
                    configDict[intArgument],
                    int, 
                    f"The {intArgument} must be an integer")


            if configDict[intArgument]>0:
                self.__dict__[intArgument] = configDict[intArgument]
        
        if 'estimators' in configDict:
            for estimatorName, estimator in configDict['estimators'].items():
                className = estimator['class']
                constructorArgs = estimator['args']
                components = className.split('.')
                module = __import__('.'.join(components[:-1]))
                for component in components[1:]:
                    module = getattr(module, component)
                classCreator = module
                if classCreator is None:
                    raise RuntimeError(f"Class {className} not found.")
 
            
                # create instance
                if constructorArgs and isinstance(constructorArgs, dict):
                    constructorArgs['baseFolder']=self.baseFolder
                    obj = classCreator(**constructorArgs)
                # elif constructorArgs and isinstance(constructorArgs, list):
                #     obj = classCreator(*constructorArgs)
                # else:
                #     obj = classCreator()
                
                self.__dict__[estimatorName] = obj
        

        

    def materialize_ensembles(self,components, ensembles):
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


    def actuate_components(self,components):
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


    def run_simulation(
        self,
        steps: int,
        components: List['Component'],
        ensembles: List['Ensemble']):
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

            materializedEnsembles = self.materialize_ensembles(components, ensembles)
            self.actuate_components(components)

            if self.stepCallback is not None:
                self.stepCallback(components, materializedEnsembles, step)


    def run_experiment(self,steps: int):

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

        for iteration in range(self.iterations):
            verbosePrint(f"Iteration {iteration + 1} started at {datetime.now()}:", 1)
            if self.prepareIteration is not None:
                self.prepareIteration(iteration)

            for simulation in range(self.simulations):
                verbosePrint(f"Simulation {simulation + 1} started at {datetime.now()}:", 2)

                components, ensembles = self.prepareSimulation(iteration, simulation)

                self.run_simulation(steps, components, ensembles)

                if self.simulationCallback is not None:
                    self.simulationCallback(components, ensembles, iteration, simulation)

            for estimator in SIMULATION_GLOBALS.estimators:
                estimator.endIteration()

            if self.iterationCallback is not None:
                self.iterationCallback(iteration)

            SIMULATION_GLOBALS.useBaselines = False
