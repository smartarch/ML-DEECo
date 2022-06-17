from ml_deeco.utils import readYaml

class Configuration:
    def __init__(self, 
                configFiles: list = [],  
                **kwargs):
        
        self.loadDefaultConfiguration()
        
        if len(configFiles) > 0:
            for f in configFiles:
                self.loadConfigurationFromFile(f)

        if len(kwargs)>0:
            for key, value in kwargs.items():
                self.set(key, value)
        
    def loadDefaultConfiguration(self):
        self.name='default'
        self.iterations = 1
        self.simulations = 1
        self.output = "output"
        self.plot = False
        self.seed = 42
        self.verbose = 0
        self.threads = 4
        self.locals = {
            'maxSteps' : 0,
        }
        self.estimators = {}

    def loadConfigurationFromFile(self, configFile):
        yaml = readYaml(configFile)
        for key, value in yaml.items():
            self.setConfig(key, value)

    def setConfig(self,key, value):
        self.__dict__[key] = value




