from ml_deeco.estimators import Estimate


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
    id : str
        Identifier of the component. Generated automatically.
    """
    id: str
    _count = 0  # Number of components of each type

    def __init__(self):
        # generate the ID
        cls = type(self)
        cls._count += 1
        self.id = "%s_%d" % (cls.__name__, cls._count)

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

    def __repr__(self) -> str:
        return self.id
