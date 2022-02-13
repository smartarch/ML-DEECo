from enum import IntEnum
from typing import Optional

from ml_deeco.simulation import Agent, Point
from ml_deeco.estimators import ValueEstimate, NeuralNetworkEstimator, TimeEstimate, NumericFeature, CategoricalFeature


class TruckState(IntEnum):
    AVAILABLE = 0          # ready to pick up a package
    LOADED = 1             # transporting a package back to the station
    MOVING_TO_STATION = 2  # returning to the station in order not to run out of fuel
    AT_STATION = 3         # inactive at the station
    TERMINATED = 4         # inactive because it ran out of fuel


class Truck(Agent):

    def __init__(self, location):
        super().__init__(location, speed=1)
        self.fuel = 1.
        self.state = TruckState.AVAILABLE
        self.target: Optional[Point] = None
        self.station = self.location

    regressionEstimate = ValueEstimate().inTimeSteps(10)\
        .using(NeuralNetworkEstimator([32], outputFolder="results/component_regression", name="Component Regression"))
    classificationEstimate = ValueEstimate().inTimeSteps(10)\
        .using(NeuralNetworkEstimator([32], outputFolder="results/component_classification", name="Component Classification"))
    timeEstimate = TimeEstimate()\
        .using(NeuralNetworkEstimator([32], outputFolder="results/component_time", name="Component Time"))

    @regressionEstimate.input(NumericFeature(0, 1))
    @classificationEstimate.input(NumericFeature(0, 1))
    @timeEstimate.input(NumericFeature(0, 1))
    def fuel(self):
        return self.fuel

    @regressionEstimate.input(CategoricalFeature(TruckState))
    @classificationEstimate.input(CategoricalFeature(TruckState))
    @timeEstimate.input(CategoricalFeature(TruckState))
    def state(self):
        return self.state

    @regressionEstimate.target(NumericFeature(0, 1))
    def fuel(self):
        return self.fuel

    @classificationEstimate.target(CategoricalFeature(TruckState))
    def state(self):
        return self.state

    @timeEstimate.condition
    def is_available(self):
        return self.state == TruckState.AVAILABLE

    @regressionEstimate.inputsValid
    @regressionEstimate.targetsValid
    @classificationEstimate.inputsValid
    @classificationEstimate.targetsValid
    @timeEstimate.inputsValid
    @timeEstimate.conditionsValid
    def not_terminated(self):
        return self.state != TruckState.TERMINATED and self.state != TruckState.AT_STATION

    def actuate(self):

        # get the values of the estimates
        fuel = self.regressionEstimate()
        assert type(fuel) == float and 0 <= fuel <= 1
        state = self.classificationEstimate()
        assert type(state) == TruckState
        time = self.timeEstimate()
        assert type(time) == float and time >= 0

        # returning to the station (without a package)
        if self.state == TruckState.MOVING_TO_STATION:
            if self.move(self.station):
                self.state = TruckState.AT_STATION

        # transporting package to the station
        elif self.state == TruckState.LOADED:
            if self.move(self.station):
                self.unload()  # we arrived at the station

        elif self.state == TruckState.AVAILABLE:
            # target location is set -- going to pick up the package
            if self.target is not None:
                if self.move(self.target):
                    self.load()

        # ran out of fuel
        if self.fuel <= 0:
            self.fuel = 0
            self.state = TruckState.TERMINATED

    def load(self):
        # print("Package loaded")
        self.target = None
        self.state = TruckState.LOADED

    def unload(self):
        # print("Package unloaded")
        self.state = TruckState.AVAILABLE

    def move(self, target):
        if self.state == TruckState.LOADED:
            self.fuel -= 0.02
        else:
            self.fuel -= 0.01
        return super().move(target)
