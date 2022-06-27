from enum import IntEnum
from typing import Optional

from ml_deeco.simulation import MovingComponent2D, Point2D
from ml_deeco.estimators import ValueEstimate, LinearRegressionEstimator

from world import WORLD


class TruckState(IntEnum):
    AVAILABLE = 0          # ready to pick up a package
    LOADED = 1             # transporting a package back to the station
    MOVING_TO_STATION = 2  # returning to the station in order not to run out of fuel
    AT_STATION = 3         # inactive at the station
    TERMINATED = 4         # inactive because it ran out of fuel


# we need to create an estimator to prepare the neural network
truckFuelEstimator = LinearRegressionEstimator(
    WORLD.experiment,
    outputFolder="results/fuel", name="Truck Fuel"
)


class Truck(MovingComponent2D):

    def __init__(self, location):
        super().__init__(location, speed=1)
        self.fuel = 1.
        self.state = TruckState.AVAILABLE
        self.target: Optional[Point2D] = None
        self.station = self.location
        self.useEstimate = False

    fuelEstimate = ValueEstimate().inTimeSteps(10).using(truckFuelEstimator)

    @fuelEstimate.input()
    @fuelEstimate.target()
    def fuel(self):
        return self.fuel

    @fuelEstimate.inputsValid
    @fuelEstimate.targetsValid
    def not_terminated(self):
        return self.state != TruckState.TERMINATED and self.state != TruckState.AT_STATION

    def actuate(self):

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
            # use the estimate to prevent running out of fuel
            if self.useEstimate:
                if self.fuelEstimate() < 0:
                    self.state = TruckState.MOVING_TO_STATION

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
