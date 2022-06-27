from ml_deeco.utils import readYaml


class Configuration:

    def __init__(self,
                 configFiles: list = [],
                 **kwargs):

        self.loadDefaultConfiguration()

        if len(configFiles) > 0:
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

    # TODO: make updates recursive -> update values in 'locals' dictionary, not replace all of them
    def loadConfigurationFromFile(self, configFile):
        yaml = readYaml(configFile)
        for key, value in yaml.items():
            self.setConfig(key, value)

    def setConfig(self, key, value):
        self.__dict__[key] = value
