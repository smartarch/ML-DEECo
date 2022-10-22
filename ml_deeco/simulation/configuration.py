import os
from typing import Union, List

from ml_deeco.utils import readYaml


class Configuration:
    """Class for loading and holding the configuration of the experiment."""

    def __init__(self, configFiles: List[Union[str, bytes, os.PathLike]] = None, **kwargs):

        self.loadDefaultConfiguration()

        if configFiles:
            for f in configFiles:
                self.loadConfigurationFromFile(f)

        if kwargs:
            for key, value in kwargs.items():
                self.setConfig(key, value)

    # noinspection PyAttributeOutsideInit
    def loadDefaultConfiguration(self):
        self.name = 'default'
        self.iterations = 1
        self.simulations = 1
        self.steps = 0
        self.output = "output"
        self.plot = False
        self.seed = 42
        self.verbose = 0
        self.threads = 4
        self.locals = {}
        self.estimators = {}

    def loadConfigurationFromFile(self, configFile: Union[str, bytes, os.PathLike]):
        """Loads configuration from a yaml file. It updates current configuration by replacing values with the same
        keys. Dictionaries are updated recursively."""
        yaml = readYaml(configFile)
        for key, value in yaml.items():
            self.setConfig(key, value)

    def setConfig(self, key, value):
        """Sets the configuration value for a specific key. If the previous value was a dictionary, it is updated recursively."""
        if key in self.__dict__:
            oldVal = self.__dict__[key]
            if isinstance(oldVal, dict) and isinstance(value, dict):
                self._nestedUpdate(oldVal, value)
                return
        if key[-7:] == ".append" and key[:-7] in self.__dict__:
            self.__dict__[key[:-7]] = self.__dict__[key[:-7]] + value
            return

        self.__dict__[key] = value

    def _nestedUpdate(self, oldDict, newDict):
        for k, v in newDict.items():
            oldVal = oldDict.get(k, None)
            if isinstance(oldVal, dict) and isinstance(v, dict):
                self._nestedUpdate(oldVal, v)
            else:
                oldDict[k] = v
