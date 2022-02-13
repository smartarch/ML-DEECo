import math
import random
from typing import Union, List, Tuple

from ml_deeco.estimators import Estimate


class Point:
    """
    Represents a location in the 2D world.

    Attributes
    ----------
    x : float
        The x-coordinate of the point.
    y : float
        The y-coordinate of the point.
    """

    def __init__(self, *coordinates: Union[float, List[float], Tuple[float, float]]):
        """
        Constructs a point from either
        x and y coordinates as two numbers (two arguments), or
        a list / tuple containing two numbers.
        """
        if len(coordinates) == 1:
            assert type(coordinates) == list or type(coordinates) == tuple
            assert len(coordinates[0]) == 2  # type: ignore
            self.x = coordinates[0][0]  # type: ignore
            self.y = coordinates[0][1]  # type: ignore
        elif len(coordinates) == 2:
            self.x = coordinates[0]
            self.y = coordinates[1]
        else:
            raise ValueError("Point must have two coordinates.")

    def __sub__(self, other):
        return self.x - other.x, self.y - other.y

    def __eq__(self, other):
        return self.x == other.x and self.y == other.y

    def __str__(self):
        return f"{self.x}, {self.y}"

    def __repr__(self):
        return f"Point({self.x}, {self.y})"

    def distance(self, other: 'Point') -> float:
        """Distance between the current point and other point."""
        dx = other.x - self.x
        dy = other.y - self.y
        dl = math.sqrt(dx * dx + dy * dy)
        return dl

    @staticmethod
    def random(x1: int, y1: int, x2: int, y2: int):
        """Returns a random point in the specified rectangular area."""
        return Point(random.randrange(x1, x2), random.randrange(y1, y2))


class ComponentMeta(type):
    """
    Metaclass for Component. Uses a counter to automatically generate the component ID.
    """

    def __new__(mcs, name, bases, namespace):
        namespace['_count'] = 0  # initialize the counter
        return super().__new__(mcs, name, bases, namespace)


class Component(metaclass=ComponentMeta):
    """
    Base class for all components.

    Attributes
    ----------
    location : Point
        The current location of the component on the 2D map.
    id : str
        Identifier of the component. Generated automatically.
    """
    location: Point
    id: str
    _count = 0  # Number of components of each type

    def __init__(self, location: Point):
        """

        Parameters
        ----------
        location : Point
            The initial location of the component.
        """
        # generate the ID
        cls = type(self)
        cls._count += 1
        self.id = "%s_%d" % (cls.__name__, cls._count)

        self.location = location

    def actuate(self):
        """
        Behavior of the component which is executed once per time step. Should be developed by the framework user.
        """
        pass

    def collectEstimatesData(self):
        """
        Collects data for Estimates. This is called from the simulation after a step is performed.
        """
        estimates = [fld for (fldName, fld) in type(self).__dict__.items()
                     if not fldName.startswith('__') and isinstance(fld, Estimate)]
        for estimate in estimates:
            estimate.collectInputs(self)
            estimate.collectTargets(self)


class Agent(Component):
    """
    Extending component with mobility.

    Attributes
    ----------
    speed : float
        The speed of the agent (movement per step).

    move(target)
        move the current location toward the target
    """

    def __init__(self, location, speed):
        """

        Parameters
        ----------
        location : Point
            The initial location of the agent.
        speed : float
            The initial speed.
        """
        Component.__init__(self, location)
        self.speed = speed

    def move(self, target: Point):
        """
        Moves the agent towards the target (based on current speed).

        Parameters
        ----------
        target : Point
            The target location.

        Returns
        -------
        bool
            True if the agent reached the target location.
        """
        dx = target.x - self.location.x
        dy = target.y - self.location.y
        dl = math.sqrt(dx * dx + dy * dy)
        if dl >= self.speed:
            self.location = Point(self.location.x + dx * self.speed / dl,
                                  self.location.y + dy * self.speed / dl)
            return False
        else:
            self.location = target
            return True
