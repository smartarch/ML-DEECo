"""
Estimator methods
"""
import abc
import os
from datetime import datetime
from typing import List
from collections import namedtuple
import numpy as np
from matplotlib import pyplot as plt
import tensorflow as tf
import seaborn as sns

from ml_deeco.estimators import Feature, CategoricalFeature, NumericFeature, BinaryFeature, TimeFeature, Estimate, BoundFeature
from ml_deeco.simulation import SIMULATION_GLOBALS
from ml_deeco.utils import Log, verbosePrint


Data = namedtuple('Data', ['x', 'y'])


#########################
# Estimator base class #
#########################


class Estimator(abc.ABC):

    def __init__(self, *, outputFolder=None, name="", skipEndIteration=False, testSplit=0.2, printLogs=True, accumulateData=False, saveCharts=True):
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
            os.makedirs(outputFolder, exist_ok=True)
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
                dataLog.export(f"{self._outputFolder}/{self._iteration}-evaluation-{label}-{targetName}.csv")

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
            plt.savefig(f"{self._outputFolder}/{self._iteration}-evaluation-{label}-{targetName}.png")
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
            plt.savefig(f"{self._outputFolder}/{self._iteration}-evaluation-{label}-{targetName}.png")
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
            plt.savefig(f"{self._outputFolder}/{self._iteration}-evaluation-{label}-{targetName}.png")
            plt.close(fig)

        return accuracy

    def endIteration(self):
        """Called at the end of the iteration. We want to do the training now."""
        self._iteration += 1

        if self._skipEndIteration:
            return

        self.collectData()

        if self._outputFolder is not None:
            self.saveData(f"{self._outputFolder}/{self._iteration}-data.csv")

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


################
# No estimator #
################


class NoEstimator(Estimator):
    """
    Does not produce any training logs or outputs. Predicts 0 for each target.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs, outputFolder=None, skipEndIteration=True, printLogs=False)

    @property
    def estimatorName(self):
        return f"NoEstimator"

    def predict(self, x):
        numTargets = sum((feature.getNumFeatures() for _, feature, _ in self._targets))
        return np.zeros([numTargets])


#################################
# Constant estimator (baseline) #
#################################


class ConstantEstimator(Estimator):
    """
    Predicts a given constant for each target.
    """

    def __init__(self, value=0., **kwargs):
        super().__init__(**kwargs)
        self._value = value

    @property
    def estimatorName(self):
        return f"ConstantEstimator({self._value})"

    def predict(self, x):
        numTargets = sum((feature.getNumFeatures() for _, feature, _ in self._targets))
        return np.full([numTargets], self._value)


###################
# Neural networks #
###################


DEFAULT_FIT_PARAMS = {
    "epochs": 50,
    "validation_split": 0.2,
    "callbacks": [tf.keras.callbacks.EarlyStopping(patience=10)],
}


class NeuralNetworkEstimator(Estimator):
    """
    Predictions based on a neural network.
    """

    @property
    def estimatorName(self):
        return f"Neural network {self._hidden_layers}"

    def __init__(self, hidden_layers, activation=None, loss=None, fit_params=None, optimizer=None, **kwargs):
        """
        Parameters
        ----------
        hidden_layers: list[int]
            Neuron counts for hidden layers.
        activation: Optional[Callable]
            Optional parameter to override the default activation function of the last layer, which is inferred from the target.
        loss: Optional[tf.keras.losses.Loss]
            Optional parameter to override the default loss function used for training, which is inferred from the target.
        fit_params: dict
            Additional parameters for the training function of the neural network (https://www.tensorflow.org/api_docs/python/tf/keras/Model#fit). The defaults are set in `DEFAULT_FIT_PARAMS`.
        optimizer: tf.optimizers.Optimizer
            Optional optimizer for the model. Default is `tf.optimizers.Adam()`.
        """
        super().__init__(**kwargs)
        self._hidden_layers = hidden_layers
        self._activation = activation
        self._optimizer = optimizer
        self._loss = loss
        self._fit_params = DEFAULT_FIT_PARAMS.copy()
        if fit_params:
            self._fit_params.update(fit_params)
        # noinspection PyTypeChecker
        self._model: tf.keras.Model = None

    def init(self, **kwargs):
        super().init(**kwargs)
        if self._model is None:
            self._model = self.constructModel()

    def constructModel(self) -> tf.keras.Model:
        numFeatures = sum((feature.getNumFeatures() for _, feature, _ in self._inputs))
        numTargets = sum((feature.getNumFeatures() for _, feature, _ in self._targets))

        if self._activation is None:
            self._activation = self.inferActivation()
        if self._loss is None:
            self._loss = self.inferLoss()

        inputs = tf.keras.layers.Input([numFeatures])
        hidden = inputs
        for layer_size in self._hidden_layers:
            hidden = tf.keras.layers.Dense(layer_size, activation=tf.keras.activations.relu)(hidden)
        output = tf.keras.layers.Dense(numTargets, activation=self._activation)(hidden)

        model = tf.keras.Model(inputs=inputs, outputs=output)

        optimizer = self._optimizer if self._optimizer else tf.optimizers.Adam()
        model.compile(
            optimizer,
            self._loss,
        )

        return model

    def inferActivation(self):
        if len(self._targets) != 1:
            raise ValueError(f"{self.name} ({self.estimatorName}): Automatic 'activation' inferring is only available for one target feature. Specify the 'activation' manually.")
        targetFeature = self._targets[0][1]
        if type(targetFeature) == Feature:
            return tf.identity
        elif type(targetFeature) == CategoricalFeature:
            return tf.keras.activations.softmax
        # NumericFeature is scaled to [0, 1], so the sigmoid ensures the correct range (which is then properly scaled in postprocess).
        elif type(targetFeature) == BinaryFeature or type(targetFeature) == NumericFeature:
            return tf.keras.activations.sigmoid
        elif type(targetFeature) == TimeFeature:
            return tf.keras.activations.exponential
        else:
            raise ValueError(f"{self.name} ({self.estimatorName}): Cannot automatically infer activation for '{type(targetFeature)}'. Specify the 'activation' manually.")

    def inferLoss(self):
        if len(self._targets) != 1:
            raise ValueError(f"{self.name} ({self.estimatorName}): Automatic 'loss' inferring is only available for one target feature. Specify the 'loss' manually.")
        targetFeature = self._targets[0][1]
        if type(targetFeature) == Feature or type(targetFeature) == NumericFeature:
            return tf.losses.MeanSquaredError()
        elif type(targetFeature) == CategoricalFeature:
            return tf.losses.CategoricalCrossentropy()
        elif type(targetFeature) == BinaryFeature:
            return tf.losses.BinaryCrossentropy()
        elif type(targetFeature) == TimeFeature:
            return tf.losses.Poisson()
        else:
            raise ValueError(f"{self.name} ({self.estimatorName}): Cannot automatically infer loss for '{type(targetFeature)}'. Specify the 'loss' manually.")

    def predict(self, x):
        return self._model(x.reshape(1, -1)).numpy()[0]

    def predictBatch(self, X):
        return self._model(X).numpy()

    def train(self, x, y):
        history = self._model.fit(
            x, y,
            **self._fit_params,
            verbose=0,
        )

        self.verbosePrint(f"Trained for {len(history.history['loss'])}/{self._fit_params['epochs']} epochs.", 2)
        self.verbosePrint(f"Training loss: {[f'{h:.4g}' for h in history.history['loss']]}", 3)
        self.verbosePrint(f"Validation loss: {[f'{h:.4g}' for h in history.history['val_loss']]}", 3)

        trainLog = Log(["epoch", "train_loss", "val_loss"])
        epochs = range(1, self._fit_params["epochs"] + 1)
        for row in zip(epochs, history.history["loss"], history.history["val_loss"]):
            trainLog.register(row)
        trainLog.export(f"{self._outputFolder}/{self._iteration}-training.csv")

    def saveModel(self, suffix=""):
        suffix = str(suffix)
        if suffix:
            filename = f"model_{suffix}.h5"
        else:
            filename = "model.h5"
        self._model.save(f"{self._outputFolder}/{filename}")

    def loadModel(self, modelPath=None):
        if modelPath is None:
            modelPath = f"{self._outputFolder}/model.h5"
        self._model = tf.keras.models.load_model(modelPath)
