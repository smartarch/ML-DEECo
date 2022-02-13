from ml_deeco.estimators import NeuralNetworkEstimator, NumericFeature, CategoricalFeature, ValueEstimate, TimeEstimate
from truck import Truck, TruckState
from ml_deeco.simulation import Ensemble, oneOf, someOf


class PackageEnsemble(Ensemble):

    def __init__(self, location):
        self.location = location  # storage location

    truck = oneOf(Truck)

    # role (ensemble-component) estimates

    truckWithRegressionEstimate = oneOf(Truck).withValueEstimate().inTimeSteps(10)\
        .using(NeuralNetworkEstimator([32], outputFolder="results/role_oneOf_regression", name="Role oneOf Regression"))
    truckWithClassificationEstimate = oneOf(Truck).withValueEstimate().inTimeSteps(10)\
        .using(NeuralNetworkEstimator([32], outputFolder="results/role_oneOf_classification", name="Role oneOf Classification"))
    truckWithTimeEstimate = oneOf(Truck).withTimeEstimate()\
        .using(NeuralNetworkEstimator([32], outputFolder="results/role_oneOf_time", name="Role oneOf Time"))

    trucksWithRegressionEstimate = someOf(Truck).withValueEstimate().inTimeSteps(10) \
        .using(NeuralNetworkEstimator([32], outputFolder="results/role_someOf_regression", name="Role someOf Regression"))
    trucksWithClassificationEstimate = someOf(Truck).withValueEstimate().inTimeSteps(10) \
        .using(NeuralNetworkEstimator([32], outputFolder="results/role_someOf_classification", name="Role someOf Classification"))
    trucksWithTimeEstimate = someOf(Truck).withTimeEstimate() \
        .using(NeuralNetworkEstimator([32], outputFolder="results/role_someOf_time", name="Role someOf Time"))

    # we use the same select for all the roles and cardinality 1 for the someOf -> all of them will select the same component

    @truck.select
    def is_available(self, truck, otherEnsembles):
        return truck.state == TruckState.AVAILABLE

    @truckWithRegressionEstimate.select
    def is_available(self, truck, otherEnsembles):
        return truck.state == TruckState.AVAILABLE

    @truckWithClassificationEstimate.select
    def is_available(self, truck, otherEnsembles):
        return truck.state == TruckState.AVAILABLE

    @truckWithTimeEstimate.select
    def is_available(self, truck, otherEnsembles):
        return truck.state == TruckState.AVAILABLE

    @trucksWithRegressionEstimate.select
    def is_available(self, truck, otherEnsembles):
        return truck.state == TruckState.AVAILABLE

    @trucksWithClassificationEstimate.select
    def is_available(self, truck, otherEnsembles):
        return truck.state == TruckState.AVAILABLE

    @trucksWithTimeEstimate.select
    def is_available(self, truck, otherEnsembles):
        return truck.state == TruckState.AVAILABLE

    @trucksWithRegressionEstimate.cardinality
    def cardinality(self):
        return 1

    @trucksWithClassificationEstimate.cardinality
    def cardinality(self):
        return 1

    @trucksWithTimeEstimate.cardinality
    def cardinality(self):
        return 1

    # inputs and outputs for the estimates

    @truckWithRegressionEstimate.estimate.input(NumericFeature(0, 1))
    @truckWithClassificationEstimate.estimate.input(NumericFeature(0, 1))
    @truckWithTimeEstimate.estimate.input(NumericFeature(0, 1))
    @trucksWithRegressionEstimate.estimate.input(NumericFeature(0, 1))
    @trucksWithClassificationEstimate.estimate.input(NumericFeature(0, 1))
    @trucksWithTimeEstimate.estimate.input(NumericFeature(0, 1))
    def fuel(self, truck):
        return truck.fuel

    @truckWithRegressionEstimate.estimate.input(CategoricalFeature(TruckState))
    @truckWithClassificationEstimate.estimate.input(CategoricalFeature(TruckState))
    @truckWithTimeEstimate.estimate.input(CategoricalFeature(TruckState))
    @trucksWithRegressionEstimate.estimate.input(CategoricalFeature(TruckState))
    @trucksWithClassificationEstimate.estimate.input(CategoricalFeature(TruckState))
    @trucksWithTimeEstimate.estimate.input(CategoricalFeature(TruckState))
    def state(self, truck):
        return truck.state

    @truckWithRegressionEstimate.estimate.target(NumericFeature(0, 1))
    @trucksWithRegressionEstimate.estimate.target(NumericFeature(0, 1))
    def fuel(self, truck):
        return truck.fuel

    @truckWithClassificationEstimate.estimate.target(CategoricalFeature(TruckState))
    @trucksWithClassificationEstimate.estimate.target(CategoricalFeature(TruckState))
    def state(self, truck):
        return truck.state

    @truckWithTimeEstimate.estimate.condition
    @trucksWithTimeEstimate.estimate.condition
    def is_available(self, truck):
        return truck.state == TruckState.AVAILABLE

    @truckWithRegressionEstimate.estimate.inputsValid
    @truckWithRegressionEstimate.estimate.targetsValid
    @truckWithClassificationEstimate.estimate.inputsValid
    @truckWithClassificationEstimate.estimate.targetsValid
    @truckWithTimeEstimate.estimate.inputsValid
    @truckWithTimeEstimate.estimate.conditionsValid
    @trucksWithRegressionEstimate.estimate.inputsValid
    @trucksWithRegressionEstimate.estimate.targetsValid
    @trucksWithClassificationEstimate.estimate.inputsValid
    @trucksWithClassificationEstimate.estimate.targetsValid
    @trucksWithTimeEstimate.estimate.inputsValid
    @trucksWithTimeEstimate.estimate.conditionsValid
    def not_terminated(self, truck):
        return truck.state != TruckState.TERMINATED and truck.state != TruckState.AT_STATION

    # ensemble estimates
    # The inputs and targets of these estimates don't make much sense as they are directly copied from the Truck component. It makes much more sense to predict values related to the ensemble itself, such as the number of components which will become members of a role (but in our case, that is not interesting as we have only cardinality 1). We also don't have any other attributes of the ensemble to use as the inputs.

    regressionEstimate = ValueEstimate().inTimeSteps(10) \
        .using(NeuralNetworkEstimator([32], outputFolder="results/ensemble_regression", name="Ensemble Regression"))
    classificationEstimate = ValueEstimate().inTimeSteps(10) \
        .using(NeuralNetworkEstimator([32], outputFolder="results/ensemble_classification", name="Ensemble Classification"))
    timeEstimate = TimeEstimate() \
        .using(NeuralNetworkEstimator([32], outputFolder="results/ensemble_time", name="Ensemble Time"))

    @regressionEstimate.input(NumericFeature(0, 1))
    @classificationEstimate.input(NumericFeature(0, 1))
    @timeEstimate.input(NumericFeature(0, 1))
    def fuel(self):
        return self.truck.fuel

    @regressionEstimate.input(CategoricalFeature(TruckState))
    @classificationEstimate.input(CategoricalFeature(TruckState))
    @timeEstimate.input(CategoricalFeature(TruckState))
    def state(self):
        return self.truck.state

    @regressionEstimate.target(NumericFeature(0, 1))
    def fuel(self):
        return self.truck.fuel

    @classificationEstimate.target(CategoricalFeature(TruckState))
    def state(self):
        return self.truck.state

    @timeEstimate.condition
    def is_available(self):
        return self.truck.state == TruckState.AVAILABLE

    @regressionEstimate.inputsValid
    @regressionEstimate.targetsValid
    @classificationEstimate.inputsValid
    @classificationEstimate.targetsValid
    @timeEstimate.inputsValid
    @timeEstimate.conditionsValid
    def not_terminated(self):
        return self.truck.state != TruckState.TERMINATED and self.truck.state != TruckState.AT_STATION

    # actuate

    def actuate(self):
        truck = self.truck

        # get the values of the estimates

        fuel = self.truckWithRegressionEstimate.estimate(truck)
        assert type(fuel) == float and 0 <= fuel <= 1
        fuel = self.trucksWithRegressionEstimate.estimate(truck)
        assert type(fuel) == float and 0 <= fuel <= 1

        state = self.truckWithClassificationEstimate.estimate(truck)
        assert type(state) == TruckState
        state = self.trucksWithClassificationEstimate.estimate(truck)
        assert type(state) == TruckState

        time = self.truckWithTimeEstimate.estimate(truck)
        assert type(time) == float and time >= 0
        time = self.trucksWithTimeEstimate.estimate(truck)
        assert type(time) == float and time >= 0

        fuel = self.regressionEstimate()
        assert type(fuel) == float and 0 <= fuel <= 1
        state = self.classificationEstimate()
        assert type(state) == TruckState
        time = self.timeEstimate()
        assert type(time) == float and time >= 0

        # the truck is available -> set its target to pick up the package
        truck.target = self.location
