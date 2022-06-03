"""
Estimator methods
"""
import abc
from datetime import datetime
from typing import List
from collections import namedtuple
import numpy as np
from matplotlib import pyplot as plt
import tensorflow as tf  # TODO: remove (and replace the usages with numpy)
import seaborn as sns

from ml_deeco.estimators import CategoricalFeature, BinaryFeature, Estimate, BoundFeature
from ml_deeco.simulation import SIMULATION_GLOBALS
from ml_deeco.utils import Log, verbosePrint

from pathlib import Path
Data = namedtuple('Data', ['x', 'y'])


#########################
# Estimator base class #
#########################


class Estimator(abc.ABC):

    def __init__(self, *, baseFolder=None, outputFolder=None, name="", skipEndIteration=False, testSplit=0.2, printLogs=True, accumulateData=False, saveCharts=True):
        """
        Parameters
        ----------
        outputFolder: Optional[str]
            The collected training data and evaluation of the training is exported there. Set to `None` to disable export.
        name: str
            String to identify the `Estimator` in the printed output of the framework (if `printLogs` is `True` and verbosity level was set by `ml_deeco.utils.setVerboseLevel`).
        skipEndIteration: bool
            Skip the training and evaluation of the model. This can be used to disable the `Estimator` temporarily while experimenting with different models.
        testSplit: float
            The fraction of the data to be used for evaluation.
        printLogs: bool
        accumulateData: bool or int
            If set to `True`, data from all previous iterations are used for training. If set to `False` (default), only the data from the last iteration are used for training. If set to an integer `k`, data from the last `k` iterations are used for training (setting this to `1` is thus equivalent to setting it to `False`).
        saveCharts: bool
            If `True`, charts are generated from the evaluation of the model.
        """
        SIMULATION_GLOBALS.estimators.append(self)

        self.data: List[Data] = []
        if outputFolder is not None:
            if baseFolder is not None and isinstance(baseFolder, Path):
                outputFolder = baseFolder / outputFolder
            else:
                outputFolder = Path() / outputFolder
            outputFolder.mkdir(parents=True, exist_ok=True)

        self._outputFolder = outputFolder
        self.name = name
        self._skipEndIteration = skipEndIteration
        self._testSplit = testSplit
        self._printLogs = printLogs
        if type(accumulateData) == bool or (type(accumulateData) == int and accumulateData >= 1):
            self._accumulateData = accumulateData
        else:
            raise ValueError("Invalid value for 'accumulateData'")
        self._saveCharts = saveCharts

        self._iteration = 0

        self._estimates: List[Estimate] = []
        self._initialized = False
        self._inputs: List[BoundFeature] = []
        self._targets: List[BoundFeature] = []

    @property
    @abc.abstractmethod
    def estimatorName(self):
        """Identification of the ML model."""
        return ""

    def assignEstimate(self, estimate: Estimate):
        self._estimates.append(estimate)

    def verbosePrint(self, message, verbosity):
        if self._printLogs:
            verbosePrint(message, verbosity)

    def init(self, force=False):
        """This must be run AFTER the input and target features are specified by the estimates."""
        if self._initialized and not force:
            self.verbosePrint(f"Already initialized {self.name} ({self.estimatorName}).", 4)
            return

        self.verbosePrint(f"Initializing Estimator {self.name} ({self.estimatorName}) with {len(self._estimates)} estimates assigned.", 1)
        if len(self._estimates) == 0:
            # print("WARNING: No Estimates assigned, the Estimator is useless.", file=sys.stderr)
            return

        for estimate in self._estimates:
            estimate.prepare()

        estimate = self._estimates[0]
        self._inputs = estimate.inputs
        self._targets = estimate.targets

        input_names = [i.name for i in self._inputs]
        target_names = [t.name for t in self._targets]

        self.verbosePrint(f"inputs {input_names}.", 2)
        self.verbosePrint(f"targets {target_names}.", 2)

        for est in self._estimates:
            assert [i.name for i in est.inputs] == input_names, f"Estimate {est} has inconsistent input features with the assigned estimator {self.name} ({self.estimatorName})"
            assert [t.name for t in est.targets] == target_names, f"Estimate {est} has inconsistent targets with the assigned estimator {self.name} ({self.estimatorName})"
            est.check()

        self._initialized = True

    def collectData(self):
        iteration_x, iteration_y = [], []
        for estimate in self._estimates:
            x, y = estimate.getData()
            iteration_x.extend(x)
            iteration_y.extend(y)

        self.verbosePrint(f"{self.name} ({self.estimatorName}): iteration {self._iteration} collected {len(iteration_x)} records.", 1)
        self.data.append(Data(iteration_x, iteration_y))

        if type(self._accumulateData) == int:
            self.data = self.data[-self._accumulateData:]
        elif not self._accumulateData:
            self.data = self.data[-1:]

    def saveData(self, fileName):
        dataLogHeader = []
        for featureName, feature, _ in self._inputs:
            dataLogHeader.extend(feature.getHeader(featureName))
        for featureName, feature, _ in self._targets:
            dataLogHeader.extend(feature.getHeader(featureName))

        dataLog = Log(dataLogHeader)

        for data_x, data_y in self.data:
            for x, y in zip(data_x, data_y):
                dataLog.register(list(x) + list(y))

        dataLog.export(fileName)

    def saveModel(self, suffix=""):
        pass

    @abc.abstractmethod
    def predict(self, x):
        """
        Parameters
        ----------
        x : np.ndarray
            Inputs, shape: [inputs]

        Returns
        -------
        np.ndarray
            Outputs, shape: [targets]
        """
        return

    def predictBatch(self, X):
        """
        Parameters
        ----------
        X : np.ndarray
            Inputs, shape: [batch_size, inputs]

        Returns
        -------
        np.ndarray
            Outputs, shape: [batch_size, targets]
        """
        return np.array([self.predict(x) for x in X])

    def train(self, X, Y):
        """
        Parameters
        ----------
        X : np.ndarray
            Inputs, shape: [batch, features].
        Y : np.ndarray
            Target outputs, shape: [batch, targets].
        """
        pass

    def evaluate(self, X, Y, label):
        """
        Parameters
        ----------
        X : np.ndarray
            Inputs, shape [batch, features].
        Y : np.ndarray
            Target outputs, shape [batch, targets].
        label : str
        """
        predictions = self.predictBatch(X)

        currentIndex = 0
        for targetName, feature, _ in self._targets:
            width = feature.getNumFeatures()
            y_pred = predictions[:, currentIndex:currentIndex + width]
            y_true = Y[:, currentIndex:currentIndex + width]
            currentIndex += width

            if width > 1:
                dataLog = Log([f"target_{i}" for i in range(width)] + [f"prediction_{i}" for i in range(width)])
            else:
                dataLog = Log(["target", "prediction"])

            for t, p in zip(y_true, y_pred):
                dataLog.register(list(t) + list(p))

            if self._outputFolder is not None:
                dataLog.export(self._outputFolder/ f"{self._iteration}-evaluation-{label}-{targetName}.csv")

            if type(feature) == BinaryFeature:
                return self.evaluate_binary_classification(label, targetName, y_pred, y_true)
            elif type(feature) == CategoricalFeature:
                return self.evaluate_classification(label, targetName, y_pred, y_true)
            else:
                return self.evaluate_regression(label, targetName, y_pred, y_true)

    def evaluate_regression(self, label, targetName, y_pred, y_true):
        mse = tf.reduce_mean(tf.metrics.mse(y_true, y_pred))
        self.verbosePrint(f"{label} – {targetName} MSE: {mse:.4g}", 2)

        if self._saveCharts and self._outputFolder is not None:
            lims = min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())
            # plt.ioff()
            fig = plt.figure(figsize=(10, 10))
            plt.axes(aspect='equal')
            plt.scatter(y_pred, y_true, alpha=0.5)
            plt.xlabel('Predictions')
            plt.ylabel('True Values')
            plt.title(f"{self.name} ({self.estimatorName})\nIteration {self._iteration}, target: {targetName}\n{label} MSE: {mse:.3f}")
            plt.xlim(lims)
            plt.ylim(lims)
            plt.plot(lims, lims, lw=0.5, c='k')
            plt.savefig(self._outputFolder / f"{self._iteration}-evaluation-{label}-{targetName}.png")
            plt.close(fig)

        return mse

    def evaluate_binary_classification(self, label, targetName, y_pred, y_true):
        accuracy = tf.reduce_mean(tf.metrics.binary_accuracy(y_true, y_pred))
        self.verbosePrint(f"{label} – {targetName} Accuracy: {accuracy:.4g}", 2)

        if self._saveCharts and self._outputFolder is not None:
            y_true = tf.squeeze(y_true)
            y_pred = tf.squeeze(y_pred > 0.5)
            cm = tf.math.confusion_matrix(y_true, y_pred)
            fig = plt.figure(figsize=(10, 10))
            sns.heatmap(cm, annot=True)
            plt.xlabel('Predictions')
            plt.ylabel('True Values')
            plt.title(f"{self.name} ({self.estimatorName})\nIteration {self._iteration}, target: {targetName}\n{label} Accuracy: {accuracy:.3f}")
            plt.savefig(self._outputFolder/ f"{self._iteration}-evaluation-{label}-{targetName}.png")
            plt.close(fig)

        return accuracy

    def evaluate_classification(self, label, targetName, y_pred, y_true):
        accuracy = tf.reduce_mean(tf.metrics.categorical_accuracy(y_true, y_pred))
        self.verbosePrint(f"{label} – {targetName} Accuracy: {accuracy:.4g}", 2)

        if self._saveCharts and self._outputFolder is not None:
            y_true_classes = tf.argmax(y_true, axis=1)
            y_pred_classes = tf.argmax(y_pred, axis=1)
            cm = tf.math.confusion_matrix(y_true_classes, y_pred_classes)
            fig = plt.figure(figsize=(10, 10))
            sns.heatmap(cm, annot=True)
            plt.xlabel('Predictions')
            plt.ylabel('True Values')
            plt.title(f"{self.name} ({self.estimatorName})\nIteration {self._iteration}, target: {targetName}\n{label} Accuracy: {accuracy:.3f}")
            plt.savefig(self._outputFolder/ f"{self._iteration}-evaluation-{label}-{targetName}.png")
            plt.close(fig)

        return accuracy

    def endIteration(self):
        """Called at the end of the iteration. We want to do the training now."""
        self._iteration += 1

        if self._skipEndIteration:
            return

        self.collectData()

        if self._outputFolder is not None:
            self.saveData(self._outputFolder/ f"{self._iteration}-data.csv")

        count = sum((len(d.x) for d in self.data))
        test_size = int(self._testSplit * count)
        if count > 0:
            x = np.concatenate([np.array(d.x) for d in self.data])
            y = np.concatenate([np.array(d.y) for d in self.data])

            if test_size > 0:
                indices = np.random.permutation(count)
                train_x = x[indices[:-test_size], :]
                train_y = y[indices[:-test_size], :]
                test_x = x[indices[-test_size:], :]
                test_y = y[indices[-test_size:], :]
            else:
                train_x = x
                train_y = y
                test_x = x[:0, :]  # empty
                test_y = y[:0, :]  # empty

            self.verbosePrint(f"{self.name} ({self.estimatorName}): Training {self._iteration} started at {datetime.now()}: ", 1)
            self.verbosePrint(f"{self.name} ({self.estimatorName}): Train data shape: {train_x.shape}, test data shape: {test_x.shape}.", 2)

            self.evaluate(train_x, train_y, label="Before-Train")
            if test_size > 0:
                self.evaluate(test_x, test_y, label="Before-Test")

            self.train(train_x, train_y)

            self.evaluate(train_x, train_y, label="Train")
            if test_size > 0:
                self.evaluate(test_x, test_y, label="Test")
