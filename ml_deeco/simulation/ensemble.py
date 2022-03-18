import operator
from collections import defaultdict
from typing import Dict, Any, TYPE_CHECKING, Union, Callable, Tuple, List, Type

from ml_deeco.estimators.estimate import TimeEstimate, ListWithEstimate, Estimate, ValueEstimate

if TYPE_CHECKING:
    from ml_deeco.estimators import Estimator
    from ml_deeco.simulation import Component


class someOf:
    """
    Declaration of a dynamic ensemble role.
    """

    # used for generating IDs
    counter = 0

    def __init__(self, compClass: Type['Component']):
        """
        Parameters
        ----------
        compClass
            Only components of this type can become members of this role.
        """
        # the id is used for sorting - the roles are evaluated in the same order they are defined
        self.id = someOf.counter
        someOf.counter += 1

        self.compClass = compClass

        self.cardinalityFn = lambda _ens: (0, None)
        self.selectFn = None
        self.utilityFn = lambda _ens, _comp: 0

        self.selections: Dict[Ensemble, Any] = defaultdict(lambda: None)

    def __get__(self, instance, owner):
        """Returns the members for the role."""
        if instance is None:
            return self
        return self.get(instance, owner)

    def get(self, instance, owner):
        """Returns the members for the role."""
        return self.selections[instance]

    def cardinality(self, cardinalityFn: Callable[['Ensemble'], Union[int, Tuple[int, int]]]):
        """
        Define the cardinality function for the role. Use this as a decorator.

        Parameters
        ----------
        cardinalityFn
            The function which returns the cardinality of the role. The returned value should be an `int` (maximum allowed number of components) or a tuple of two `int`s (minimum, maximum). Both minimum and maximum are inclusive.
        """
        self.cardinalityFn = cardinalityFn
        return self

    def select(self, selectFn: Callable[['Ensemble', 'Component', List['Ensemble']], bool]):
        """
        Define the select predicate for the role. Use this as a decorator.

        Parameters
        ----------
        selectFn
            The select predicate of the role. The parameters of the function are:
                - the ensemble instance (self),
                - the component instance (only components of type `compClass` specified in the `__init__` are considered),
                - the list of already materialized ensembles.
            If the predicate returns `True`, the component can be selected for the role.
        """
        self.selectFn = selectFn
        return self

    def utility(self, utilityFn: Callable[['Ensemble', 'Component'], float]):
        """
        Define the utility function for the role. Use this as a decorator.

        Parameters
        ----------
        utilityFn
            The utility function of the components for role. The parameters of the function are:
                - the ensemble instance (self),
                - the component instance (only components which passed the `select` are considered).
            During materialization, the components are ordered by the descending utility (components with the biggest utility are selected).
        """
        self.utilityFn = utilityFn
        return self

    # region Helper functions for the role members selection

    def reset(self, instance):
        self.selections[instance] = None

    def filterComponentsByType(self, instance, allComponents):
        return [comp for comp in allComponents if
                isinstance(comp, self.compClass)]

    def filterPreviouslySelectedComponents(self, instance, components):
        return [comp for comp in components if
                comp not in self.selections[instance]]

    def filterBySelectFunction(self, instance, components, otherEnsembles):
        return [comp for comp in components if
                self.selectFn(instance, comp, otherEnsembles)]

    def assignUtility(self, instance, components):
        return [(self.utilityFn(instance, comp), comp) for comp in components]

    def selectComponents(self, instance, allComponents, otherEnsembles):
        filteredComponents = self.filterComponentsByType(instance, allComponents)
        filteredComponents = self.filterPreviouslySelectedComponents(instance, filteredComponents)
        filteredComponents = self.filterBySelectFunction(instance, filteredComponents, otherEnsembles)
        return self.assignUtility(instance, filteredComponents)

    # endregion

    def execute(self, instance, allComponents, otherEnsembles):
        """
        Performs the role members selection.

        Parameters
        ----------
        instance : Ensemble
            The ensemble instance.
        allComponents : List[Component]
            All components in the system.
        otherEnsembles : List[Ensemble]
            Already materialized ensembles.

        Returns
        -------
        bool
            True if role members were selected.
        """

        assert (self.cardinalityFn is not None)
        assert (self.selectFn is not None)

        self.selections[instance] = []

        # get the cardinality
        cardinality = self.cardinalityFn(instance)
        if isinstance(cardinality, tuple):
            cardinalityMin, cardinalityMax = cardinality
            if cardinalityMax is None:
                cardinalityMax = len(allComponents)
        else:
            cardinalityMin, cardinalityMax = cardinality, cardinality

        # perform the selection
        sel = self.selectComponents(instance, allComponents, otherEnsembles)
        for idx in range(cardinalityMax):
            if len(sel) > 0:
                utility, comp = max(sel, key=operator.itemgetter(0))
                self.selections[instance].append(comp)
                sel = self.selectComponents(instance, allComponents, otherEnsembles)
            else:
                break

        if len(self.selections[instance]) < cardinalityMin:
            return False

        return True

    def withValueEstimate(self, collectOnlyIfMaterialized=True):
        """Assign a `ValueEstimate` to the role."""
        return someOfWithEstimate(self.compClass, ValueEstimate(), collectOnlyIfMaterialized)

    def withTimeEstimate(self, collectOnlyIfMaterialized=True, **dataCollectorKwargs):
        """Assign a `TimeEstimate` to the role."""
        return someOfWithEstimate(self.compClass, TimeEstimate(**dataCollectorKwargs), collectOnlyIfMaterialized)


class someOfWithEstimate(someOf):

    def __init__(self, compClass, estimate: 'Estimate', collectOnlyIfMaterialized: bool):
        super().__init__(compClass)
        self.estimate = estimate
        self.estimate.inputsIdFunction = lambda instance, comp: comp
        self.estimate.targetsIdFunction = self.estimate.inputsIdFunction
        self.estimate.collectOnlyIfMaterialized = collectOnlyIfMaterialized

    def using(self, estimator: 'Estimator'):
        """Assigns an estimator to the estimate."""
        self.estimate.using(estimator)
        return self

    def inTimeSteps(self, timeSteps):
        """Automatically collect the data with fixed time difference between inputs and targets."""
        self.estimate.inTimeSteps(timeSteps)
        return self

    def get(self, instance, owner):
        sel = super().get(instance, owner)
        if isinstance(sel, list):
            sel = ListWithEstimate(sel)

        def estimate(*args, ignoreCache=False):
            return self.estimate.estimate(instance, *args, ignoreCache=ignoreCache)

        sel.estimate = estimate
        return sel

    def collectEstimateData(self, instance, allComponents, materialized):
        if self.estimate.collectOnlyIfMaterialized and not materialized:
            return

        filteredComponents = self.filterComponentsByType(instance, allComponents)
        for comp in filteredComponents:
            self.estimate.collectInputs(instance, comp)
            self.estimate.collectTargets(instance, comp)

    def selectComponents(self, instance, allComponents, otherEnsembles):
        """Estimate caching for better performance."""
        filteredComponents = self.filterComponentsByType(instance, allComponents)
        self.estimate.cacheEstimates(instance, filteredComponents)
        filteredComponents = self.filterPreviouslySelectedComponents(instance, filteredComponents)
        filteredComponents = self.filterBySelectFunction(instance, filteredComponents, otherEnsembles)
        return self.assignUtility(instance, filteredComponents)


class oneOf(someOf):
    def __init__(self, compClass):
        super().__init__(compClass)
        self.cardinalityFn = lambda inst: 1

    def get(self, instance, owner):
        sel = super().get(instance, owner)
        return sel[0]

    def cardinality(self, cardinalityFn):
        raise TypeError("To specify cardinality, use 'someOf' instead of 'oneOf'.")

    def withValueEstimate(self, collectOnlyIfMaterialized=True):
        """Assign a `ValueEstimate` to the role."""
        return oneOfWithEstimate(self.compClass, ValueEstimate(), collectOnlyIfMaterialized)

    def withTimeEstimate(self, collectOnlyIfMaterialized=True, **dataCollectorKwargs):
        """Assign a `TimeEstimate` to the role."""
        return oneOfWithEstimate(self.compClass, TimeEstimate(**dataCollectorKwargs), collectOnlyIfMaterialized)


class oneOfWithEstimate(someOfWithEstimate):
    def __init__(self, compClass, estimate: 'Estimate', collectOnlyIfMaterialized: bool):
        if hasattr(compClass, "estimate"):
            raise TypeError(f"The component type '{self.compClass}' cannot be used with 'oneOfWithEstimate' as it already has another attribute named 'estimate'. Please rename the attribute 'estimate' in '{self.compClass}'.")
        super().__init__(compClass, estimate, collectOnlyIfMaterialized)
        self.cardinalityFn = lambda inst: 1

    def get(self, instance, owner):
        sel = super().get(instance, owner)
        selected = sel[0]

        def estimate(*args, ignoreCache=False):
            return self.estimate.estimate(instance, *args, ignoreCache=ignoreCache)

        selected.estimate = estimate
        return selected

    def cardinality(self, cardinalityFn):
        raise TypeError("To specify cardinality, use 'someOf' instead of 'oneOf'.")


class Ensemble:

    def __init__(self):
        self.materialized = False

    def materialize(self, components, otherEnsembles):
        """
        Performs the ensemble materialization.

        Parameters
        ----------
        components : List[Component]
            All components in the system.
        otherEnsembles : List[Ensemble]
            Already materialized ensembles.

        Returns
        -------
        bool
            True if the ensemble was materialized.
        """
        self.materialized = False

        # sort the roles of the ensemble according to id
        # note: depending on the name of the select and cardinality functions, we might get duplicates here which we definitely don't want
        compFields = sorted(set([fld for (fldName, fld) in type(self).__dict__.items()
                                 if not fldName.startswith('__') and isinstance(fld, someOf)]),
                            key=lambda fld: fld.id)

        for fld in compFields:
            fld.reset(self)
        if not self.situation():
            return

        # select members for the roles
        allOk = True
        for fld in compFields:
            if not fld.execute(self, components, otherEnsembles):
                allOk = False
                break

        if not allOk:
            for fld in compFields:
                fld.reset(self)
        else:
            self.materialized = True

        return allOk

    def situation(self) -> bool:
        """Assesses whether this ensemble instance can be materialized in this time step."""
        return True

    def actuate(self):
        """The function performed when the ensemble is materialized. To be implemented by the user."""
        pass

    def priority(self) -> float:
        """Priority of the ensemble. Ensembles with higher priority get materialized earlier."""
        return 1

    def __lt__(self, other):
        return self.priority() > other.priority()

    def collectEstimatesData(self, components):
        """
        Collects data for Estimates assigned to ensembles and ensemble roles. This is called from the simulation after a step is performed.
        """
        # ensemble roles
        rolesWithEstimate = [fld for (fldName, fld) in type(self).__dict__.items()
                             if not fldName.startswith('__') and isinstance(fld, someOfWithEstimate)]
        # note: depending on the name of the 'select', 'utility' and 'cardinality' functions, we might get duplicates here which we definitely don't want
        for role in set(rolesWithEstimate):
            role.collectEstimateData(self, components, self.materialized)

        # ensemble
        if not self.materialized:
            return  # do not collect data when ensemble is not materialized
        estimates = [fld for (fldName, fld) in type(self).__dict__.items()
                     if not fldName.startswith('__') and isinstance(fld, Estimate)]
        for estimate in estimates:
            estimate.collectInputs(self)
            estimate.collectTargets(self)
