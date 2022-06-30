# Simple example of ML-DEECo usage

This aims to be a simple minimal example which shows how to use ML-DEECo. It assumes that the reader has already read the [README file](../../../ml_deeco/README.md) of the `ml_deeco` package.

The example models a package transportation use case. We assume to have a truck which can pick up packages and transport them to a station. To simplify the case, the truck will only move along one route with the station on one end and a package storage on the other end.

The example requires Python 3 with `numpy`, `sklearn`, `matplotlib` and `seaborn` libraries installed. Furthermore, the `ml_deeco` package must be installed (see [README file of `ml_deeco`](../../../ml_deeco/README.md#installation)).

## Running the example

The example can be run by executing the [`run.py`](run.py) file after `ml_deeco` has been installed (see [README file](../../../ml_deeco/README.md)).

It runs two simulations of the example &ndash; one to collect the data for the ML model, and a second one to use the trained model during the simulation.

## Implementation

### Definition of Truck component

The truck is modeled as a component (based on `ml_deeco.simulation.Agent` class as it is capable of movement) in [`truck.py`](truck.py). We assume the truck has a limited amount of fuel and the fuel consumption depends on whether the truck is loaded with package or not.

We use a machine learning model (linear regression) to predict the fuel level of the truck in the future &ndash; after 10 steps of the simulation. If the model predicts that the truck will run out of fuel in the 10 steps, the truck becomes unavailable and returns to the station to get there safely and not run out of fuel while transporting a package.

The estimate is assigned to the component with input and target specified by the decorators:
```py
fuelEstimate = ValueEstimate().inTimeSteps(10).using('truckFuelEstimator')

@fuelEstimate.input()
@fuelEstimate.target()
def fuel(self):
    return self.fuel
```

Notice that the `'truckFuelEstimator'` is assigned to the estimate using a string identifier. The Estimator (ML model) is created inside the `TruckExperiment` in the [`run.py`](run.py) and assigned to the estimate using the `initEstimates` method (see section [Simulation](#simulation)).

We also add guards to prevent collecting data when the truck is inactive:
```py
@fuelEstimate.inputsValid
@fuelEstimate.targetsValid
def not_terminated(self):
    return self.state != TruckState.TERMINATED and self.state != TruckState.AT_STATION
```

### Definition of Package ensemble

We use an ensemble to assign a job to the truck &ndash; [`package_ensemble.py`](package_ensemble.py). If the truck is available, the ensemble orders it to go pick up a package. When the truck is loaded, it transports the package to the station and becomes available again. We assume there are enough packages in the storage, so the ensemble will assign another package once the truck is available again.

### Simulation

The simulation is managed by the `TruckExperiment` class (derived from the `ml_deeco.simulation.Experiment` class).

When creating the `TruckExperiment` object, we set the configuration to two iterations of running the simulation with ML model training in between iterations. We only run the simulation once in each iteration, but running it more times is useful for collecting more data for training the ML model. The truck component and package ensemble for each simulation run are created in the `prepareSimulation` method.

## Results

If we compare the logs from the two runs, we can see that in the first iteration, when the ML model was not used, the truck ran out of fuel and was terminated. In the second iteration, with the ML model, the system predicted that the truck will run out of fuel and decided to order it to go back to the station.

Please note that these results depend on the random initialization of the ML model. If a different random seed is chosen, the model might not train well. This is caused by using too few data for training, and can be solved by increasing the number of simulations (to, e.g., 5 or 10) in each iteration.
