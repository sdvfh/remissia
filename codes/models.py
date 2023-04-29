import numpy as np
import tensorflow as tf
import xgboost as xgb
from constants import dataset_types, patience
from cuml import LogisticRegression, RandomForestClassifier
from cuml.internals.memory_utils import set_global_output_type
from tensorflow.python.keras.callbacks import EarlyStopping
from tensorflow.python.keras.layers import Conv1D, Dense, Dropout, Flatten, MaxPooling1D
from tensorflow.python.keras.models import Sequential

set_global_output_type("numpy")


class Model:
    name = None

    def __init__(self, orc):
        """The __init__ function is called when the class is instantiated. It
        sets up the object with all of its attributes and methods. The __init__
        function takes in a single argument, orc, which should be an instance
        of Orc.

        Parameters
        ----------
            self
                Represent the instance of the class
            orc
                Pass in the orc object

        Returns
        -------

            Nothing

        Doc Author
        ----------
            Trelent
        """
        self.orc = orc
        self.datasets = {dataset_type: None for dataset_type in dataset_types}
        self.args = {}
        self.hypers = {}
        self.predictions = None
        return

    def run(self):
        """The run function is the main function of this class. It does the
        following: 1) Processes the dataset, which includes loading it and
        splitting it into train/test sets. 2) Processes hyperparameters, which
        includes setting them to default values if they are not provided by
        user input. 3) Trains a model using these hyperparameters on this
        dataset and saves its results in a dictionary called 'results'.

        Parameters
        ----------
            self
                Represent the instance of the class

        Returns
        -------

            Nothing

        Doc Author
        ----------
            Trelent
        """
        self._process_dataset()
        self._process_hyperparameters()
        self._train_model()
        self._save_results()
        return

    def _process_dataset(self):
        """The _process_dataset function is a helper function that takes the
        dataset_types list and iterates through it. It then assigns each of the
        datasets to their respective keys in the self.datasets dictionary.

        Parameters
        ----------
            self
                Refer to the instance of the class

        Returns
        -------

            The datasets dictionary

        Doc Author
        ----------
            Trelent
        """
        for dataset_type in dataset_types:
            dataset = self.orc.datasets[dataset_type]
            self.datasets[dataset_type] = dataset
        return

    def _process_hyperparameters(self):
        """The _process_hyperparameters function is used to process the
        hyperparameters of a model. It takes in the hyperparameters as an
        argument and returns them after processing. The function can be used to
        do any preprocessing on the hyperparameters before they are passed into
        a model.

        Parameters
        ----------
            self
                Represent the instance of the class

        Returns
        -------

            A dictionary of hyperparameters

        Doc Author
        ----------
            Trelent
        """
        return

    def _train_model(self):
        """The _train_model function is used to train the model. It takes no
        arguments and returns nothing.

        Parameters
        ----------
            self
                Represent the instance of the class

        Returns
        -------

            Nothing, so the model is not trained

        Doc Author
        ----------
            Trelent
        """
        return

    def _save_results(self):
        """The _save_results function is called by the _fit function. It saves
        the predictions, hyperparameters, and arguments to a dictionary in
        self.orc.results[self.name]. The name of this dictionary is set by
        self._set_name().

        Parameters
        ----------
            self
                Refer to the instance of the class

        Returns
        -------

            A dictionary of the predictions, hyperparameters, and arguments

        Doc Author
        ----------
            Trelent
        """
        self.orc.results[self.name] = {
            "predictions": self.predictions.round().copy(),
            "hypers": self.hypers.copy(),
            "args": self.args.copy(),
        }
        return


class XGBoost(Model):
    name = "XGBoost"
    _initial_args = {
        "eta": 0.3,
        "gamma": 0.0,
        "max_depth": 6,
        "min_child_weight": 1.0,
        "max_delta_step": 0.0,
        "subsample": 1.0,
        "sampling_method": "gradient_based",
        "colsample_bytree": 1.0,
        "colsample_bylevel": 1.0,
        "colsample_bynode": 1.0,
        "lambda": 1.0,
        "alpha": 0.0,
        "tree_method": "gpu_hist",
        "sketch_eps": 0.03,
        "scale_pos_weight": 1.0,
        "refresh_leaf": 1,
        "process_type": "default",
        "grow_policy": "depthwise",
        "max_leaves": 0,
        "max_bin": 256,
        "objective": "binary:logistic",
        "seed_per_iteration": True,
        "verbosity": 0,
    }

    def _process_dataset(self):
        """The _process_dataset function takes the attributes and target from
        the orc object, and creates a dictionary of xgb.DMatrix objects for
        each dataset type (train, test, etc.)

        Parameters
        ----------
            self
                Refer to the object itself

        Returns
        -------

            Nothing, as it is a void function

        Doc Author
        ----------
            Trelent
        """
        target = self.orc.target
        attributes = self.orc.attributes
        datasets = self.orc.datasets

        for dataset_type in dataset_types:
            dataset = datasets[dataset_type]
            df_attributes, df_target = dataset[attributes], dataset[target]
            self.datasets[dataset_type] = xgb.DMatrix(df_attributes, df_target)
        return

    def _process_hyperparameters(self):
        """The _process_hyperparameters function is used to set the
        hyperparameters of the model.

        Parameters
        ----------
            self
                Access the attributes and methods of the class

        Returns
        -------

            Nothing

        Doc Author
        ----------
            Trelent
        """
        target = self.orc.target
        self.args = self._initial_args.copy()
        train_y = self.orc.datasets["train"][target]
        if self.orc.problem.name == "classification":
            scale_pos_weight = (train_y == 0).sum() / (train_y == 1).sum()
            self.args["scale_pos_weight"] = scale_pos_weight
        return

    def _train_model(self):
        """The _train_model function is the main function of the class. It
        takes in a dictionary of arguments, a train set, validation set and
        test set. The function then trains an XGBoost model on the training
        data and validates it on the validation data using early stopping with
        a patience value defined above. The best number of trees is saved as
        well as scale_pos_weight which was passed into this class from the
        hyperparameter optimization step.

        Parameters
        ----------
            self
                Bind the method to the object

        Returns
        -------

            Nothing

        Doc Author
        ----------
            Trelent
        """
        args = self.args
        train = self.datasets["train"]
        valid = self.datasets["valid"]
        train_valid = self.datasets["train_valid"]
        test = self.datasets["test"]

        args.update({"seed": self.orc.seed})

        ml = xgb.train(
            params=args,
            dtrain=train,
            num_boost_round=1,
            early_stopping_rounds=patience,
            verbose_eval=100,
            evals=[(train, "Train"), (valid, "Validation")],
        )
        self.hypers["best_ntree_limit"] = ml.best_ntree_limit
        self.hypers["scale_pos_weight"] = self.args["scale_pos_weight"]

        ml = xgb.train(
            params=args,
            dtrain=train_valid,
            num_boost_round=self.hypers["best_ntree_limit"],
            verbose_eval=100,
            evals=[(train_valid, "Train_Valid")],
        )
        self.predictions = ml.predict(test)
        return


class Regression(Model):
    name = "Logistic Regression"

    def _process_hyperparameters(self):
        """
        The _process_hyperparameters function is used to modify the hyperparameters dictionary
            before it is passed to the model. This function can be used for a variety of purposes,
            including:

        Parameters
        ----------
            self
                Access the orc object

        Returns
        -------

            A dictionary of hyperparameters

        Doc Author
        ----------
            Trelent
        """
        target = self.orc.target
        train_y = self.orc.datasets["train"][target]
        if self.orc.problem.name == "classification":
            class_weight = train_y.count() / (2 * np.bincount(train_y))
            class_weight = dict(zip(range(2), class_weight))
            self.args["class_weight"] = class_weight
        return

    def _train_model(self):
        """The _train_model function is the main function of the model. It
        should take in a dataset and return a trained model. The _train_model
        function must be defined for all models, but it can be as simple or
        complex as needed.

        Parameters
        ----------
            self
                Access the class attributes and methods

        Returns
        -------

            Nothing

        Doc Author
        ----------
            Trelent
        """
        target = self.orc.target
        attributes = self.orc.attributes
        args = self.args
        train_valid = self.datasets["train_valid"]
        test = self.datasets["test"]

        ml = LogisticRegression(
            class_weight=args["class_weight"],
            max_iter=1,
        )
        ml.fit(train_valid[attributes].astype(np.float32), train_valid[target])
        self.predictions = ml.predict_proba(test[attributes].astype(np.float32))[:, 0]
        return


class RandomForest(Model):
    name = "Random Forest"

    def _train_model(self):
        """The _train_model function is the main function of the class. It
        takes in a dataset and trains a model on it. The _train_model function
        should be able to take in any dataset, as long as it has been
        preprocessed by the preprocess_data method.

        Parameters
        ----------
            self
                Access the attributes and methods of the class

        Returns
        -------

            Nothing, but it does set the predictions attribute of the object

        Doc Author
        ----------
            Trelent
        """
        target = self.orc.target
        attributes = self.orc.attributes
        train_valid = self.datasets["train_valid"]
        test = self.datasets["test"]
        seed = self.orc.seed

        ml = RandomForestClassifier(random_state=seed, n_streams=1)
        ml.fit(train_valid[attributes].astype(np.float32), train_valid[target])
        self.predictions = ml.predict_proba(test[attributes].astype(np.float32))[:, 0]
        return


class NeuralNetwork(Model):
    name = "Neural Network"
    classification_params = {"activation": "sigmoid", "loss": "BinaryCrossentropy"}
    regressions_params = {"activation": "linear", "loss": "MSE"}

    def __init__(self, *args, **kwargs):
        """The __init__ function is called when the class is instantiated. It
        sets up the attributes of an instance of a class. The __init__ function
        can take arguments, but self must be the first one.

        Parameters
        ----------
            self
                Represent the instance of the class
            *args
                Pass a non-keyworded, variable-length argument list to the function
            **kwargs
                Pass keyworded, variable-length argument list to a function

        Returns
        -------

            Nothing

        Doc Author
        ----------
            Trelent
        """
        super().__init__(*args, **kwargs)
        physical_devices = tf.config.list_physical_devices("GPU")
        tf.config.experimental.set_memory_growth(physical_devices[0], True)
        tf.random.set_seed(seed=self.orc.seed)
        return

    def _train_model(self):
        """The _train_model function is the main function of the class. It
        trains a neural network model on the training data and validates it on
        the validation data. The best epoch is determined by looking at the
        validation loss, and then that number of epochs are used to train a new
        model using both training and validation datasets combined. Finally,
        predictions are made for all test samples.

        Parameters
        ----------
            self
                Access the attributes and methods of the class

        Returns
        -------

            Nothing

        Doc Author
        ----------
            Trelent
        """
        train = self.datasets["train"]
        valid = self.datasets["valid"]
        train_valid = self.datasets["train_valid"]
        test = self.datasets["test"]
        batch_size = 16
        verbose = 2
        params = self.orc.problem.get_neural_network_params()

        ml, callbacks = self._create_model(params)
        history = ml.fit(
            x=train["attributes"],
            y=train["target"],
            validation_data=(valid["attributes"], valid["target"]),
            batch_size=batch_size,
            epochs=1,
            verbose=verbose,
            callbacks=callbacks,
        )

        best_value = self.orc.problem.search_best(history.history["val_loss"])
        best_epoch = history.history["val_loss"].index(best_value) + 1
        self.hypers["epochs"] = best_epoch
        ml, _ = self._create_model(params)
        ml.fit(
            x=train_valid["attributes"],
            y=train_valid["target"],
            batch_size=batch_size,
            epochs=self.hypers["epochs"],
            verbose=verbose,
        )
        self.predictions = ml.predict(test["attributes"])[:, 0]
        return

    def _create_model(self, params):
        return []


class MLP(NeuralNetwork):
    name = "Multilayer Perceptron"

    def _process_dataset(self):
        """The _process_dataset function takes the attributes and target from
        the original dataset, and creates a new dictionary with two keys:
        &quot;attributes&quot; and &quot;target&quot;. The values of these keys
        are numpy arrays containing all of the attribute data (X) and target
        data (y). This dictionary is then stored in self.datasets[dataset_type]
        for each dataset type.

        Parameters
        ----------
            self
                Bind the method to the class

        Returns
        -------

            A dictionary with the keys 'attributes' and 'target'

        Doc Author
        ----------
            Trelent
        """
        attributes = self.orc.attributes
        target = self.orc.target
        for dataset_type in dataset_types:
            dataset = self.orc.datasets[dataset_type]
            self.datasets[dataset_type] = {
                "attributes": dataset[attributes].values,
                "target": dataset[target].values,
            }
        return

    def _create_model(self, params):
        """The _create_model function is a helper function that creates the
        model. It takes in a dictionary of parameters and returns the model, as
        well as any callbacks to be used during training. The _create_model
        function is called by KerasClassifier or KerasRegressor when creating
        their models.

        Parameters
        ----------
            self
                Bind the instance of the class to a method
            params
                Pass the parameters to the model

        Returns
        -------

            A tuple of a model and a list of callbacks

        Doc Author
        ----------
            Trelent
        """
        early = EarlyStopping(patience=patience, restore_best_weights=True)
        model = Sequential()
        model.add(Dense(100, activation="relu", input_dim=len(self.orc.attributes)))
        model.add(Dropout(0.5))
        model.add(Dense(50, activation="relu"))
        model.add(Dropout(0.5))
        model.add(Dense(10, activation="relu"))
        model.add(Dense(1, activation=params["activation"]))
        model.compile(loss=params["loss"], optimizer="adam")
        return model, [early]


class CNN(NeuralNetwork):
    name = "Convolutional Neural Network"

    def _process_dataset(self):
        """The _process_dataset function is used to process the data in the
        datasets dictionary. The function takes no arguments and returns
        nothing. The function uses a for loop to iterate through each dataset
        type, and then creates a new dictionary entry with keys
        &quot;attributes&quot; and &quot;target&quot;. The value of the
        attributes key is an array of all of the attribute values from that
        dataset, while target contains an array of all of its target values.
        These arrays are expanded by one dimension so that they can be fed into
        Keras models.

        Parameters
        ----------
            self
                Bind the method to an object

        Returns
        -------

            A dictionary with the keys 'attributes' and 'target'

        Doc Author
        ----------
            Trelent
        """
        attributes = self.orc.attributes
        target = self.orc.target
        for dataset_type in dataset_types:
            dataset = self.orc.datasets[dataset_type]
            self.datasets[dataset_type] = {
                "attributes": np.expand_dims(dataset[attributes].values, axis=-1),
                "target": np.expand_dims(dataset[target].values, axis=-1),
            }
        return

    def _create_model(self, params):
        """The _create_model function is a function that creates the model. It
        takes in one parameter, params, which is a dictionary of
        hyperparameters. The _create_model function returns two values: the
        model and an array of callbacks to be used during training.

        Parameters
        ----------
            self
                Bind the method to an object
            params
                Pass the parameters to the model

        Returns
        -------

            A tuple of the model and a list of callbacks

        Doc Author
        ----------
            Trelent
        """
        early = EarlyStopping(patience=patience, restore_best_weights=True)
        model = Sequential()
        model.add(
            Conv1D(
                filters=256,
                kernel_size=4,
                activation="relu",
                input_shape=(len(self.orc.attributes), 1),
            )
        )
        model.add(Conv1D(filters=64, kernel_size=4, activation="relu"))
        model.add(Dropout(0.5))
        model.add(MaxPooling1D(pool_size=2))
        model.add(Flatten())
        model.add(Dense(200, activation="relu"))
        model.add(Dense(1, activation=params["activation"]))
        model.compile(loss=params["loss"], optimizer="adam")
        return model, [early]


models = {
    Regression.name: Regression,
    RandomForest.name: RandomForest,
    XGBoost.name: XGBoost,
    MLP.name: MLP,
    CNN.name: CNN,
}
