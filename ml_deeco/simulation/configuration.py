import os
from typing import Union, List

from ml_deeco.utils import readYaml


class Configuration:

    def __init__(self,
                 configFiles: List[Union[str, bytes, os.PathLike]] = None,
                 **kwargs):

        self.loadDefaultConfiguration()

        if configFiles and len(configFiles) > 0:
            for f in configFiles:
                self.loadConfigurationFromFile(f)

        if len(kwargs) > 0:
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

        self.__dict__[key] = value

    def _nestedUpdate(self, oldDict, newDict):
        for k, v in newDict.items():
            oldVal = oldDict.get(k, None)
            if isinstance(oldVal, dict) and isinstance(v, dict):
                self._nestedUpdate(oldVal, v)
            else:
                oldDict[k] = v
