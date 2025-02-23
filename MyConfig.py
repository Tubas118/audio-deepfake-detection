import os
import yaml

class MyConfig:

    def __init__(self, configFilename):
        self.configFilename = configFilename

        with open("config.yml", "r") as f:
            configLoader = yaml.safe_load(f.read())

        self.activeDataSource = configLoader['active-data-source']
        self.vendor = configLoader['data-sources'][self.activeDataSource]['vendor']

        rawDataPath = configLoader['data-sources'][self.activeDataSource]['dataPath']
        expandedDataPath = os.path.expandvars(rawDataPath)
        self.dataPath = expandedDataPath.replace("\\", "/")

        self.modelName = configLoader['model']['name']

    def fullPath(self, appendPath):
        return os.path.join(self.dataPath, appendPath)