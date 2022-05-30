import tensorflow as tf

from ml_deeco.estimators import Estimator, Feature, CategoricalFeature, BinaryFeature, NumericFeature, TimeFeature
from ml_deeco.utils import Log

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
