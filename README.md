# ML-DEECo

ML-DEECo is a machine-learning-enabled component model for adaptive component architectures. It is based on DEECo component model, which features autonomic *components* and dynamic component coalitions (called *ensembles*). ML-DEECo allows exploiting machine learning in decisions about adapting component coalitions at runtime. 

The framework uses neural networks trained in a supervised manner. A simulation of the system is run to collect data used for training the neural network. The simulation can then be run again with the trained model to see the impact of the learned model on the system.

## Contents

* [Installation](#installation)
* [Usage](#usage)
  * [Specifying components](#specifying-components)
  * [Specifying ensembles](#specifying-ensembles)
  * [Adding machine-learning-based estimates](#adding-machine-learning-based-estimates)
  * [Running the simulation](#running-the-simulation)
* [Notes to implementation](#notes-to-implementation)

## Installation

Use `pip`:

```
pip install .
```

For local development, installing with `--editable` is recommended:

```
pip install --editable .
```

## Usage

The ML-DEECo framework provides abstractions for creating components and ensembles and assigning machine learning estimates to them. A simulation can then be run with the components and ensembles to observe behavior of the system and collect data for training the estimates. The trained estimate can then be used in the next run of the simulation. 

### Specifying components

The `ml_deeco.simulation` module offers base classes for `Component` and `Agent` (components with movement, the `Agent` class is derived from `Component`).

Each component has a `location` in a 2D space defined by an instance of `ml_deeco.simulation.Point` class. The component has an `actuate` method which is executed in every step of the simulation and should be implemented by the user.

The `Agent` offers the `move` method which will move the component in a direction towards a target defined in the parameter. If the agent arrives at the target, the `move` method will return `True`.

```py
from ml_deeco.simulation import Component, Agent


# Example of a stationary component -- a charging station
class Charger(Component):

    def __init__(self, location):
        super().__init__(location)
        self.charging_drones = []

    # a drone at the location of the charger can start charging
    def startCharging(self, drone):
        if drone.location == self.location:
            self.charging_drones.append(drone)

    def actuate(self):
        # we charge the drones
        for drone in self.charging_drones:
            drone.battery += 0.01
            if drone.battery == 1:
                # fully charged
                drone.station = None


# Example of an agent -- moving component
class Drone(Agent):

    def __init__(self, location, speed):
        super().__init__(location, speed)
        self.battery = 1
        self.charger = None

    def actuate(self):
        # if the drone has an assigned charger
        if self.charger:
            # fly towards it
            if self.move(self.charger.location):
                # drone arrived at the location of the charger
                self.charger.startCharging(self)

```

### Specifying ensembles

Ensembles are meant for coordination of the components. The base class for ensembles is `ml_deeco.simulation.Ensemble`. Each ensemble has a priority specified by overriding the `priority` method. Furthermore, ensemble can contain components in static and dynamic roles. Static roles are represented simply as variables of the ensemble. 

The declaration of a dynamic role is done via the `someOf` (meaning a list of components) or `oneOf` (single component) function with the component type as an argument. The components are assigned and re-assigned to the dynamic roles (we say that a component becomes a member of the ensemble) by the framework in every step of the simulation. To select the members for the role, several conditions can be specified using decorators:

* `select` is a predicate, which the component must pass to be picked;
* `utility` orders the components;
* `cardinality` sets the maximum (or both minimum and maximum) allowed number of components to be picked.

The member selection works by first finding all components of the correct type that pass the `select` predicate, then ordering them by the `utility` (higher utility is better) and using the `cardinality` to limit the number of selected members. The `cardinality` can also be used to limit the minimal number of member components &ndash; if there are not enough components passing the selection, the ensemble cannot be active at the time.

```py
from ml_deeco.simulation import Ensemble, someOf


class ChargingAssignment(Ensemble):
    # static role
    charger: Charger

    # dynamic role
    drones: List[Drone] = someOf(Drone)

    # we select those drones which need charging
    @drones.select
    def need_charging(self, drone, otherEnsembles):
        return drone.needs_charging

    # order them by the missing battery (so the drones with less battery are selected first)
    @drones.utility
    def missing_battery(self, drone):
        return 1 - drone.battery

    # and limit the cardinality to the number of free slots on the charger
    @drones.cardinality
    def free_slots(self):
        return 0, self.charger.free_slots

    def __init__(self, charger):
        self.charger = charger

    def actuate(self):
        # assign the charger to each drone -- it will start flying towards it to charge
        for drone in self.drones:
            drone.station = self.charger
```

The framework performs ensemble materialization (selection of the ensembles which should be active at this time) in every step of the simulation. The ensembles are materialized in a greedy fashion, ordered by their priority (descending). Only those ensembles for which all roles were assigned appropriate number of members (conforming to the cardinality) can be materialized. For all materialized ensembles, the `actuate` method is called.

### Adding machine-learning-based estimates

There are two types of tasks our framework focuses on &ndash; value estimate and time-to-condition estimate. 

In the value estimate, we use the currently available observations to predict some value that can be observed only at some future point (after a fixed amount of time steps). Based on the type of the estimated value, the supervised ML models are usually divided into regression and classification.

The time-to-condition estimates focuses on predicting how long it will take until some condition will become true. This is done by specifying a condition over some future values of component fields.

The tasks are described in more detail in the paper.

The definition of each estimate is split to three parts:

1. The definition of the `Estimator` &ndash; machine learning model and storage for the collected data. 
2. The declaration of the `Estimate` field in the component or ensemble.
3. The definition of inputs, target and guards. These are realized as decorators on component/ensemble fields and getter functions of the component. 

All of these steps are realized using the `ml_deeco.estimators` module.

#### Estimator

Estimator represents the underlying machine learning model for computing the estimates. The framework features `ConstantEstimator` and `NeuralNetworkEstimator`.

Common parameters for the initializer of the `Estimator`s are:
- `outputFolder` &ndash; The collected training data and evaluation of the training is exported there. Set to `None` to disable export.
- `name` &ndash; String to identify the `Estimator` in the printed output of the framework (if `printLogs` is `True` and verbosity level was set by `ml_deeco.utils.setVerboseLevel`).
- `accumulateData` &ndash; If set to `True`, data from all previous iterations are used for training. If set to `False` (default), only the data from the last iteration are used for training.

The `ConstantEstimator` is initialized with a constant, which is then returned every time predictions are requested. It can serve as a baseline in experiments.

The `NeuralNetworkEstimator` uses [TensorFlow framework](https://www.tensorflow.org/) to implement a feedforward neural network. It is enough to specify the number of neurons in hidden layers using the `hidden_layers` parameter. The model is constructed and trained appropriate to the target feature specified by the `Estimate`.

```py
from ml_deeco.estimators import NeuralNetworkEstimator

futureBatteryEstimator = NeuralNetworkEstimator(
    hidden_layers=[256, 256],  # two hidden layers
    outputFolder="results/drone_battery", name="Drone Battery"
)
```

#### Adding the estimate

##### To a component

The estimate is created by initializing the `ValueEstimate` class (future value estimate &ndash; both regression and classification) or `TimeEstimate` (time-to-condition estimate) and assigned as class variables of the component (in fact, they are implemented as properties).

In case of value estimate, the number of time steps we want to predict into the future is set using the `inTimeSteps` method.

For both `Estimate` and `TimeEstimate`, the `Estimator` (described in the previous section) must be assigned. That is done by the `using` method.

Multiple estimates can be assigned to a component.

```py
from ml_deeco.estimators import ValueEstimate

class Drone(Agent):

    futureBatteryEstimate = ValueEstimate().inTimeSteps(50)\
        .using(futureBatteryEstimator)  # defined earlier
    
    # more code of the component
```

##### To an ensemble

The estimates can be added to ensembles in a same way as to components &ndash; as class variables (properties).

##### To an ensemble role (ensemble-component pair)

To assign an estimate to a role, use the `withEstimate` (value estimate) or `withTimeEstimate` (time-to-condition estimate) methods of `someOf` (or `oneOf`). Only one estimate can be assigned to a role.

The `Estimator` must be also assigned by the `using` method. In case of value estimate, the number of time steps we want to predict into the future is set using the `inTimeSteps` method.

```py
from ml_deeco.simulation import Ensemble, someOf

waitingTimeEstimator = NeuralNetworkEstimator(
    hidden_layers=[256, 256],  # two hidden layers
    outputFolder="results/waiting_time", name="Waiting time"
)

class DroneChargingAssignment(Ensemble):

    # dynamic role with time estimate
    drones: List[Drone] = someOf(Drone).withTimeEstimate()\
                                       .using(waitingTimeEstimator)
    
    # more code of the ensemble
```

#### Configuring inputs, target and guards

The definition of inputs, target and guards is realized as decorators and getter functions. For estimates assigned to components and ensembles, the decorator has a syntax `@<estimateName>.<configuration>`. For estimates assigned to roles, the syntax is `@<roleName>.estimate.<configuration>`.

The decorators are applied to methods of the component or ensemble. For estimates assigned to components and ensembles, these methods should only have the `self` parameter. For estimates assigned to roles, these methods are expected to have the `self` parameter and a second parameter representing a component (the potential role member).

##### Inputs

The inputs of the estimate are specified using the `input()` decorator, optionally with a feature type as a parameter. We offer a `FloatFeature(min, max)`, which performs normalization of the inputs, a `CategoricalFeature(enum|list)` for one-hot encoding categorical values, and a `BinaryFeature()` to represent boolean attributes.

Example of inputs for an estimate in a component (continued from earlier):

```py
from ml_deeco.estimators import ValueEstimate, NumericFeature, CategoricalFeature
from ml_deeco.simulation import Agent

class Drone(Agent):

    # create the estimate (as described earlier)
    futureBatteryEstimate = ValueEstimate().inTimeSteps(50)\
        .using(futureBatteryEstimator)
    
    def __init__(self, location):
        self.battery = 1
        self.state = DroneState.IDLE
        # more code

    # numeric feature
    @futureBatteryEstimate.input(NumericFeature(0, 1))
    def battery(self):
        return self.battery

    # categorical feature constructed from an enum
    @futureBatteryEstimate.input(CategoricalFeature(DroneState))
    def drone_state(self):
        return self.state
```

Example of input for an estimate connected to a role (continued from earlier):

```py
class DroneChargingAssignment(Ensemble):

    # dynamic role with time estimate (as described earlier)
    drones: List[Drone] = someOf(Drone).withTimeEstimate()\
                                       .using(waitingTimeEstimator)
    
    @drones.estimate.input(NumericFeature(0, 1))
    def battery(self, drone):
        return drone.battery
```

##### Target for `ValueEstimate`

The target is specified similarly to the inputs using `target()` decorator. A `Feature` can again be given as a parameter &ndash; this is how classification and regression tasks are distinguished. The feature is then used to set the appropriate number of neurons and the activation function of the last layer of the neural network and the loss function used for training (more details in [Notes to implementation](#notes-to-implementation)).

```py
class Drone(Agent):

    # create the estimate and inputs as described earlier
    ...

    # define the target -- regression task
    @futureBatteryEstimate.target(NumericFeature(0, 1))
    def battery(self):
        return self.battery
```

##### Condition for `TimeEstimate`

For the time-to-condition estimate, a condition must be specified instead of the target value. The syntax is again similar &ndash; using the `condition` decorator. If multiple conditions are provided, they are considered in an "and" manner. 

```py
class DroneChargingAssignment(Ensemble):

    # create the estimate and inputs as described earlier
    ...
    
    # define the condition (drone is accepted for charging)
    @drones.estimate.condition
    def is_accepted(self, drone):
        return drone in self.charger.acceptedDrones
```

##### Validity of inputs &ndash; guards

Guard functions can be specified using `inputsValid`, `targetsValid` and `conditionValid` decorators to assess the validity of inputs and targets. The data are collected for training only if the guard conditions are satisfied. This can be used for example to prevent collecting data from components which are no longer active.

```py
class Drone(Agent):

    # create the estimate, inputs and targets as described earlier
    ...
    
    @futureBatteryEstimate.inputsValid
    @futureBatteryEstimate.targetsValid
    def not_terminated(self):
        return self.state != DroneState.TERMINATED
```

#### Obtaining the estimated value

The `Estimate` object is callable, so the value of the estimate based on the current inputs can be obtained by calling the estimate as a function. For estimate assigned to a role, a component instance is expected as an argument of the call.

Example in a component:

```py
class Drone(Agent):

    # create the estimate, inputs and targets as described earlier
    ...
    
    def actuate(self):
        estimatedFutureBattery = self.futureBatteryEstimate()
```

Example for a role:

```py
class DroneChargingAssignment(Ensemble):

    # create the estimate and inputs as described earlier
    drones: List[Drone] = someOf(Drone).withTimeEstimate()\
                                       .using(waitingTimeEstimator)
    ...
    
    @drones.select
    def drones(self, drone, otherEnsembles):
        # we obtain the estimated waiting time here
        waitingTime = self.drones.estimate(drone)
        # and use it to decide whether the drone should ask for a charging slot 
        return drone.needsCharging(waitingTime)
```

### Running the simulation

The `ml_deeco.simulation` module offers two functions for running the simulation: `run_simulation` and `run_experiment`.

#### `run_simulation`

The `run_simulation(components, ensembles, steps)` function runs the simulation with `components` and `ensembles` for `steps` steps. An optional `stepCallback` can be supplied which is called after each simulation step. It can be used for example to log data from the simulation. The parameters are:
- list of all components in the system,
- list of materialized ensembles (in this time step),
- current time step (int).

Before running the simulation, the `Estimator`s have to be initialized. The easiest way to do that is by calling the `SIMULATION_GLOBALS.initEstimators()`.

#### `run_experiment`

The `run_experiment` is useful for running the simulation several times with training of the ML models in between. The `iterations` parameter specifies the number of iterations. In each iteration, the simulation is run `simulation` times. After that, the data from all the simulations in the current iteration are used to train the ML model (`Estimator`). The next iteration will use the updated model.

The `prepareSimulation` function is used to obtain the components and ensembles for the simulation (it gets the current iteration and the current simulation as parameters). The simulation is then run using our `run_simulation` function for `steps` steps.

The `prepareIteration` is an optional function to be run at the beginning of each iteration. It can be used for example to initialize logs for logging data during simulations. Apart from the `stepCallback`, we also allow specifying a `simulationCallback` (ran after each simulation) and `iterationCallback` (ran at the end of iteration after the ML training finished).

The initialization of the `Estimator`s is done automatically in the `run_experiment` function.

#### Running the simulation manually

For better control over the simulation, one can also run the simulation loop manually. The functions `materialize_ensembles` and `actuate_components` can be useful for that (they are used inside our `run_simulation`).

## Notes to implementation

### Construction of NN models

We use the target feature to automatically infer the activation function for the last layer of the neural network and the loss function for training.

| Feature                                 | Last layer activation                | Loss                      |
|-----------------------------------------|--------------------------------------|---------------------------|
| `Feature` (default)                     | identity                             | Mean squared error        |
| `NumericFeature`                        | sigmoid (+ scaling to proper range)  | Mean squared error        |
| `CategoricalFeature`                    | softmax (1 neuron for each category) | Categorical cross-entropy |
| `BinaryFeature`                         | sigmoid (only 1 neuron)              | Binary cross-entropy      |
| `TimeFeature` (used by `TimeEstimate`)  | exponential                          | Poisson                   |

### Caching of estimates

For role-assigned estimates, we compute the estimated values for all potential member components at the same time and cache them. It saves time as the neural network is capable of processing all the potential members in one batch. This implies that the inputs of the model can't use the information about the already selected members for this role.
