import os
import yaml

class MyConfig:

    def __init__(self, configFilename):
        self.configFilename = configFilename

        with open("config.yml", "r") as f:
            configLoader = yaml.safe_load(f.read())

        # Read active data source
        self.activeDataSource = configLoader['active-data-source']

        self.vendor = configLoader['data-sources'][self.activeDataSource]['vendor']
        rawDataPath = configLoader['data-sources'][self.activeDataSource]['dataPath']
        expandedDataPath = os.path.expandvars(rawDataPath)
        self.dataPath = expandedDataPath.replace("\\", "/")

        # Read active job
        self.activeJob = configLoader['active-job']
        activeJobParameters = configLoader['jobs'][self.activeJob]

        self.trainDatasetPath = activeJobParameters['train-dataset-path']
        self.trainLabelFilePath = activeJobParameters['train-label-file-path']
        self.numClasses = activeJobParameters['num-classes']
        self.testDatasetPath = activeJobParameters['test-dataset-path']
        self.modelName = activeJobParameters['model-name']
        self.sampleRate = activeJobParameters['sample-rate']
        self.duration = activeJobParameters['duration']
        self.numMels = activeJobParameters['num-mels']
        self.maxTimeSteps = activeJobParameters['max-time-steps']

    def fullPath(self, appendPath):
        return os.path.join(self.dataPath, appendPath)