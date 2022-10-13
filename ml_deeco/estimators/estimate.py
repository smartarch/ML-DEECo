"""
Estimates
"""
import abc
from collections import namedtuple
from enum import Enum
from typing import Callable, List, TYPE_CHECKING, Optional, Dict, Union
import numpy as np

from ml_deeco.estimators import Feature, TimeFeature, NumericFeature

if TYPE_CHECKING:
    from ml_deeco.estimators import Estimator


BoundFeature = namedtuple('BoundFeature', ('name', 'feature', 'function'))


class DataCollectorMode(Enum):
    First = 1  # keep only the first record for each recordId
    Last = 2   # keep only the last record for each recordId
    All = 3    # keep all records for each recordId


class DataCollector:
    """Data storage for the training data to match them by `recordId`."""

    def __init__(self, estimate, begin=DataCollectorMode.All):
        self.estimate = estimate
        self._records = {}
        self._begin = begin
        self.x = []
        self.y = []

    def collectRecordInputs(self, recordId, inputs, extra=None, force_replace=False):
        """Collects inputs for a given `recordId`."""
        if recordId not in self._records or force_replace:
            self._records[recordId] = []

        records = self._records[recordId]
        if self._begin == DataCollectorMode.All:
            records.append((inputs, extra))
        elif self._begin == DataCollectorMode.First:
            if len(records) == 0:
                records.append((inputs, extra))
        elif self._begin == DataCollectorMode.Last:
            if len(records) == 0:
                records.append((inputs, extra))
            else:
                records[0] = (inputs, extra)

    def collectRecordTargets(self, recordId, targets, recordGuards, *args):
        """Collects targets (true values) for a given `recordId`."""
        if recordId not in self._records:
            # the record with corresponding ID doesn't exist, the data probably weren't valid at the time
            return

        records = self._records[recordId]
        del self._records[recordId]

        callable_targets = {name: target for name, target in targets.items() if callable(target)}

        for inputs, extra in records:
            for name in callable_targets:
                targets[name] = callable_targets[name](inputs, extra)

            for guard in recordGuards:
                guardParamCount = guard.__code__.co_argcount
                guardParams = [*args, inputs, targets, extra][:guardParamCount]
                if not guard(*guardParams):
                    return

            x = self.estimate.preprocessInputs(inputs)
            y = self.estimate.preprocessTargets(targets)

            self.x.append(x)
            self.y.append(y)

    def clear(self):
        self.x = []
        self.y = []
        self._records = {}


class Estimate(abc.ABC):
    """
    Base class for ValueEstimate and TimeEstimate.
    """

    def __init__(self, **dataCollectorKwargs):
        # noinspection PyTypeChecker
        self.estimator: 'Estimator' = None

        self.inputs: List[BoundFeature] = []
        self.extras: List[BoundFeature] = []
        self.targets: List[BoundFeature] = []

        self.inputsIdFunction = lambda *args: (*args,)
        self.targetsIdFunction = lambda *args: (*args,)
        self.inputsGuards: List[Callable] = []
        self.targetsGuards: List[Callable] = []
        self.recordGuards: List[Callable] = []

        self.baseline = lambda *args: self.estimator.baseline
        self.dataCollector = DataCollector(self, **dataCollectorKwargs)
        self.estimateCache: Dict[Dict] = dict()  # used only for estimates assigned to roles

    def linkExperiment(self, experiment):
        """Assigns an estimator from the experiment. This is necessary when the estimator was specified as a string in
        the 'using' method."""
        from ml_deeco.estimators import Estimator
        if isinstance(self.estimator, str):
            if not hasattr(experiment, self.estimator) or \
                    not isinstance(getattr(experiment, self.estimator), Estimator):
                raise AttributeError(f"Estimator '{self.estimator}' not found in the experiment class.")
            self.estimator = getattr(experiment, self.estimator)
            self.estimator.assignEstimate(self)

    def using(self, estimator: Union['Estimator', str]):
        """Assigns an estimator to the estimate. If the `estimator` is specified as a string, the `linkExperiment`
        method must be called before using the estimator."""
        from ml_deeco.estimators import Estimator
        self.estimator = estimator
        if isinstance(estimator, Estimator):
            estimator.assignEstimate(self)
        return self

    def withBaseline(self, baseline: Callable):
        """Set the baseline function which is used before the first training."""
        self.baseline = baseline
        return self

    @abc.abstractmethod
    def prepare(self):
        """The prepare function is called after all the decorators have initialized the inputs, targets, etc. and can be
        used to modify them."""
        pass

    def check(self):
        """Checks whether the estimate is initialized properly."""
        assert self.estimator is not None, "No estimator assigned, use the 'using' method to assign an estimator."
        assert self.inputsIdFunction is not None, f"{self.estimator.name}: 'inputsId' function not specified."
        assert self.targetsIdFunction is not None, f"{self.estimator.name}: 'targetsId' function not specified."
        assert len(self.inputs) > 0, f"{self.estimator.name}: No inputs specified."
        assert len(self.targets) > 0, f"{self.estimator.name}: No targets specified."

    # decorators (definition of inputs and outputs)

    def input(self, feature: Optional[Feature] = None):
        """Defines an input feature."""
        if feature is None:
            feature = Feature()

        def addInputFunction(function):
            self._addInput(function.__name__, feature, function)
            return function

        return addInputFunction

    def extra(self, function):
        """
        Defines an extra input feature – not given to the prediction model.

        We use this for example to save the time of the inputs in the time-to-condition estimate so that we can compute
        the time difference.
        """
        self._addExtra(function.__name__, function)
        return function

    def inputsId(self, function):
        """Defines the function for matching the inputs with according targets. Unless there is a specific need for
        modifying the default behavior, do not use this decorator."""
        self.inputsIdFunction = function
        return function

    def targetsId(self, function):
        """Defines the function for matching the targets with according inputs. Unless there is a specific need for
        modifying the default behavior, do not use this decorator."""
        self.targetsIdFunction = function
        return function

    def inputsValid(self, function):
        """Guard for detecting whether the inputs are valid and can be used for training. Use this as a decorator."""
        self.inputsGuards.append(function)
        return function

    def recordValid(self, function):
        """Guard for detecting whether the combination of inputs and outputs are valid and can be used for training.
        Use this as a decorator."""
        self.recordGuards.append(function)
        return function

    def _addInput(self, name: str, feature: Feature, function: Callable):
        self.inputs.append(BoundFeature(name, feature, function))

    def _addExtra(self, name: str, function: Callable):
        self.extras.append(BoundFeature(name, None, function))

    def _addTarget(self, name: str, feature: Feature, function: Callable):
        self.targets.append(BoundFeature(name, feature, function))

    # estimation

    def _estimate(self, *args):
        """Helper function to compute the estimate."""
        if self.estimator.experiment.useBaselines:
            return self.baseline(*args)

        x = self.generateRecord(*args)
        prediction = self.estimator.predict(x)
        return self.generateOutputs(prediction)

    def estimate(self, *args, ignoreCache=False):
        """
        Computes the estimate (based on the current values of the attributes).

        Parameters
        ----------
        ignoreCache : bool
            Set to True to ignore the cached values (which are used only for role-assigned estimates) and compute the
            estimate again.

        Returns
        -------
        prediction : Any
            The predicted value (if there is only one target), or a dictionary `{ feature_name: predicted_value }` with
            all targets.
        """
        if self.estimateCache and not ignoreCache:  # the cache is non-empty
            ensemble, comp = args
            if comp in self.estimateCache[ensemble]:
                return self.estimateCache[ensemble][comp]
            else:
                return self._estimate(*args)

        return self._estimate(*args)

    def cacheEstimates(self, ensemble, components):
        """Computes the estimates for all components in a batch at the same time and caches the results. Use this only
        for estimates assigned to ensemble roles."""
        if self.estimator.experiment.useBaselines:
            self.estimateCache[ensemble] = {
                comp: self.baseline(ensemble, comp)
                for comp in components
            }
        else:
            records = np.array([self.generateRecord(ensemble, comp) for comp in components])

            predictions = self.estimator.predictBatch(records)

            self.estimateCache[ensemble] = {
                comp: self.generateOutputs(prediction)
                for comp, prediction in zip(components, predictions)
            }

    def generateInputs(self, *args) -> Dict:
        """Generates the inputs record."""
        return {
            name: function(*args)
            for name, _, function in self.inputs
            if function is not None
        }

    def preprocessInputs(self, inputs: Dict) -> np.ndarray:
        """Preprocesses the inputs record for the `Estimator.predict` function."""
        record = []
        for name, feature, _ in self.inputs:
            if name not in inputs:
                continue
            value = inputs[name]
            value = feature.preprocess(value)
            record.append(value)
        return np.concatenate(record)

    def generateRecord(self, *args):
        return self.preprocessInputs(self.generateInputs(*args))

    def generateOutputs(self, prediction):
        """Generates the outputs from the `Estimator.predict` prediction."""

        # if we have only one target, return just the value
        if len(self.targets) == 1:
            return self.targets[0][1].postprocess(prediction)

        # otherwise, return a dictionary with all the targets
        output = {}
        currentIndex = 0
        for name, feature, _ in self.targets:
            width = feature.getNumFeatures()
            values = prediction[currentIndex:currentIndex + width]
            output[name] = feature.postprocess(values)
            currentIndex += width

        return output

    def __get__(self, instance, owner):
        if instance is None:
            return self

        def estimate(*args):
            return self.estimate(instance, *args)

        return estimate

    # data collection

    def collectInputs(self, *args, inputs=None, id=None):
        """Collects the inputs for training."""
        for f in self.inputsGuards:
            if not f(*args):
                return

        if inputs is None:
            inputs = self.generateInputs(*args)

        extra = {
            name: function(*args)
            for name, _, function in self.extras
        }

        recordId = id if id is not None else self.inputsIdFunction(*args)
        self.dataCollector.collectRecordInputs(recordId, inputs, extra)

    def generateTargets(self, *args) -> Dict:
        """Generates the targets record."""
        return {
            name: function(*args)
            for name, _, function in self.targets
        }

    def preprocessTargets(self, targets: Dict) -> np.ndarray:
        """Preprocesses the targets record for the training of `Estimator`."""
        record = []
        for name, feature, _ in self.targets:
            value = targets[name]
            value = feature.preprocess(value)
            record.append(value)
        return np.concatenate(record)

    def collectTargets(self, *args, id=None):
        """Collects the targets for training."""
        for f in self.targetsGuards:
            if not f(*args):
                return

        targets = self.generateTargets(*args)

        recordId = id if id is not None else self.targetsIdFunction(*args)
        self.dataCollector.collectRecordTargets(recordId, targets, self.recordGuards, *args)

    def getData(self, clear=True):
        """Gets (and optionally clears) all collected data."""
        x, y = self.dataCollector.x, self.dataCollector.y
        if clear:
            self.dataCollector.clear()
        return x, y


class ValueEstimate(Estimate, abc.ABC):
    """
    Implementation of the value estimate (both regression and classification). Predicts a future value based on current
    observations.
    """

    def __init__(self):
        super().__init__()

    def prepare(self):
        # nothing needed here
        pass

    def inTimeSteps(self, timeSteps):
        """Automatically collect data with fixed time difference between inputs and targets."""
        self.targetsGuards.append(lambda *args: self.estimator.experiment.currentTimeStep >= timeSteps)
        self.inputsIdFunction = lambda *args: (*args, self.estimator.experiment.currentTimeStep)
        self.targetsIdFunction = lambda *args: (*args, self.estimator.experiment.currentTimeStep - timeSteps)
        return self

    def inTimeStepsRange(self, minTimeSteps, maxTimeSteps, trainingPercentage=1):
        """
        Automatically collect data with a variable time difference (from a specified interval) between inputs and
        targets.

        Parameters
        ----------
        minTimeSteps : int
            Minimal allowed difference of time steps for collecting data.
        maxTimeSteps : int
            Maximal allowed difference of time steps for collecting data.
        trainingPercentage : float
            Percentage (between 0 and 1) of the collected data to use for training. If it is lower than 1, a step bigger
            than 1 is used to select only part of the data for training.
        """
        self.__class__ = ValueEstimateRange  # use the ValueEstimateRange implementation instead of ValueEstimate

        timeStepsStep = int(max(1 // trainingPercentage, 1))
        # noinspection PyAttributeOutsideInit
        self.timeStepsRange = (minTimeSteps, maxTimeSteps, timeStepsStep)

        self.inputs.insert(0, BoundFeature("time", NumericFeature(minTimeSteps, maxTimeSteps), None))
        self.targetsGuards.append(lambda *args: self.estimator.experiment.currentTimeStep >= minTimeSteps)
        self.inputsIdFunction = lambda *args, time: (*args, self.estimator.experiment.currentTimeStep + time)
        self.targetsIdFunction = lambda *args: (*args, self.estimator.experiment.currentTimeStep)
        return self

    def target(self, feature: Optional[Feature] = None):
        """Defines a target value."""
        if feature is None:
            feature = Feature()

        def addTargetFunction(function):
            self._addTarget(function.__name__, feature, function)
            return function

        return addTargetFunction

    def targetsValid(self, function):
        """Guard for detecting whether the targets are valid and can be used for training. Use this as a decorator."""
        self.targetsGuards.append(function)
        return function


class ValueEstimateRange(ValueEstimate):
    """
    Implementation of the value estimate (both regression and classification) with a range of time differences between
    inputs and outputs. This class should not be used directly and should only be obtained using the
    'ValueEstimate().inTimeStepsRange(...)' call.
    """

    # noinspection PyMissingConstructor
    def __init__(self):
        raise SyntaxError("Use 'ValueEstimate().inTimeStepsRange(...)' instead.")

    def collectInputs(self, *args, **kwargs):
        """Collects the inputs for training."""
        for f in self.inputsGuards:
            if not f(*args):
                return

        for time in range(*self.timeStepsRange):
            recordId = self.inputsIdFunction(*args, time=time)
            inputs = self.generateInputs(*args, time)

            extra = {
                name: function(*args)
                for name, _, function in self.extras
            }

            self.dataCollector.collectRecordInputs(recordId, inputs, extra)

    def generateInputs(self, *args):
        """Generates the inputs record."""
        *args, time = args
        inputs = super().generateInputs(*args)
        inputs["time"] = time
        return inputs

    def cacheEstimates(self, ensemble, components):
        pass  # caching is not applicable for time steps range


class TimeEstimate(Estimate):
    """
    Implementation of the time-to-condition estimate.
    """

    def __init__(self, **dataCollectorKwargs):
        super().__init__(**dataCollectorKwargs)
        self.timeFunc = self.time(lambda *args: self.estimator.experiment.currentTimeStep)

        self.targets = [BoundFeature("time", TimeFeature(), None)]
        self.conditionFunctions = []

    def prepare(self):
        # The conditions work the same way as targets guards (false == invalid data), but we want to perform them after
        # all the guards passed.
        self.targetsGuards += self.conditionFunctions

    def time(self, function):
        """Defines how to measure time for the time-to-condition estimate. The default uses the current time step of the
        simulation, so if the simulation is run using our `Experiment` class, there is no need to overriding the default
        behavior using this function."""
        self.timeFunc = function
        self.extras = [BoundFeature("time", TimeFeature(), self.timeFunc)]
        return function

    def condition(self, function):
        """Defines the condition for the time-to-condition estimate. If multiple conditions are defined, they are
        considered in an "and" manner."""
        self.conditionFunctions.append(function)
        return function

    def conditionsValid(self, function):
        """Guard for detecting whether the conditions are valid and can be used for training. Use this as a
        decorator."""
        self.targetsGuards.append(function)
        return function

    def generateTargets(self, *args):
        currentTimeStep = self.timeFunc(*args)

        def timeDifference(inputs, extra):
            difference = currentTimeStep - extra['time']
            return difference

        return {
            "time": timeDifference
        }


class ListWithEstimate(list):
    estimate: Optional[Estimate] = None
