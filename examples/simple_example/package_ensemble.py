from truck import Truck, TruckState
from ml_deeco.simulation import Ensemble, oneOf


class PackageEnsemble(Ensemble):

    def __init__(self, location):
        super().__init__()
        self.location = location  # storage location

    truck = oneOf(Truck)

    @truck.select
    def is_available(self, truck, otherEnsembles):
        return truck.state == TruckState.AVAILABLE

    def actuate(self):
        # the truck is available -> set its target to pick up the package
        self.truck.target = self.location
