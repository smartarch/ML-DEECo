import math
import random
from typing import Union, List, Tuple

from ml_deeco.simulation import Component


class Point2D:
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
            raise ValueError("Point2D must have two coordinates.")

    def __sub__(self, other):
        return self.x - other.x, self.y - other.y

    def __eq__(self, other):
        return self.x == other.x and self.y == other.y

    def __str__(self):
        return f"{self.x}, {self.y}"

    def __repr__(self):
        return f"Point2D({self.x}, {self.y})"

    def distance(self, other: 'Point2D') -> float:
        """Distance between the current point and other point."""
        dx = other.x - self.x
        dy = other.y - self.y
        dl = math.sqrt(dx * dx + dy * dy)
        return dl

    @staticmethod
    def random(left: int, top: int, right: int, bottom: int):
        """Returns a random point in the specified rectangular area."""
        return Point2D(random.randrange(left, right), random.randrange(top, bottom))


class StationaryComponent2D(Component):
    """
    Stationary component with a location on 2D map.

    Attributes
    ----------
    location : Point2D
        The location of the component on the 2D map.
    """
    location: Point2D

    def __init__(self, location: Point2D):
        """
        Parameters
        ----------
        location : Point2D
            The initial location of the component.
        """
        super().__init__()
        self.location: Point2D = location


class MovingComponent2D(StationaryComponent2D):
    """
    Extending component with mobility.

    Attributes
    ----------
    speed : float
        The speed of the agent (movement per step).
    """

    def __init__(self, location, speed=1):
        """
        Parameters
        ----------
        location : Point2D
            The initial location of the agent.
        speed : float
            The initial speed.
        """
        super().__init__(location)
        self.speed = speed

    def move(self, target: Point2D):
        """
        Moves the agent towards the target (based on current speed).

        Parameters
        ----------
        target : Point2D
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
            self.location = Point2D(self.location.x + dx * self.speed / dl,
                                    self.location.y + dy * self.speed / dl)
            return False
        else:
            self.location = target
            return True
