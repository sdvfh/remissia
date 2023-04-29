import gzip
import pickle

from models import NeuralNetwork, models
from scipy.stats import mode

from sklearn.metrics import (
    balanced_accuracy_score,
    cohen_kappa_score,
    f1_score,
    precision_score,
    recall_score,
)

class _Problem:
    name = None
    _metrics = []
    search_best = None

    def __init__(self, orc):
        """
    The __init__ function is called when the class is instantiated.
    It sets up the object with all of its instance variables and methods.
    The self parameter refers to the object itself, and allows you to access or modify any of its attributes.

    Parameters
    ----------
        self
            Represent the instance of the class
        orc
            Set the orc attribute of the object

    Returns
    -------

        Nothing

    Doc Author
    ----------
        Trelent
    """
        self.orc = orc
        return

    @staticmethod
    def get_stratification(target):
        """
    The get_stratification function takes in a target and returns the stratification of that target.
        The stratification is determined by the number of unique values in the column, as well as whether or not it is numeric.
        If there are less than 10 unique values, then it will be considered categorical. Otherwise, if there are more than 10
            unique values and all of them can be converted to floats (i.e., they're numeric), then it will be considered continuous.

    Parameters
    ----------
        target
            Specify the column that we want to stratify on

    Returns
    -------

        A list of the unique values in a column

    Doc Author
    ----------
        Trelent
    """
        return

    def evaluate(self):
        """
    The evaluate function is used to evaluate the performance of a model.
    It takes in two arguments:
        1) The dataframe containing the features and labels for training/testing,
            and
        2) A boolean value indicating whether or not we are evaluating on test data.

    Parameters
    ----------
        self
            Represent the instance of the class

    Returns
    -------

        The value of the expression

    Doc Author
    ----------
        Trelent
    """
        return

    def get_neural_network_params(self):
        """
    The get_neural_network_params function is used to return a dictionary of parameters that are
    used to construct the neural network. The keys in this dictionary should be the same as those
    that are used by your neural network class's constructor.

    Parameters
    ----------
        self
            Represent the instance of the class

    Returns
    -------

        A dictionary of the parameters

    Doc Author
    ----------
        Trelent
    """
        return {}

    def save_checkpoint(self):
        """
    The save_checkpoint function saves the results of a simulation to disk.

    Parameters
    ----------
        self
            Bind the object to the method

    Returns
    -------

        Nothing

    Doc Author
    ----------
        Trelent
    """
        with gzip.open(
            self.orc.path["tumor_group"] / f"{self.orc.seed}.gz.pkl", "wb"
        ) as file:
            pickle.dump(self.orc.results, file)
        return

    def checkpoint_exists(self):
        """
    The checkpoint_exists function checks to see if a checkpoint file exists for the current seed.
    If it does, then we can skip running the simulation and just load in the results from that file.

    Parameters
    ----------
        self
            Bind the object to the class

    Returns
    -------

        A boolean value

    Doc Author
    ----------
        Trelent
    """
        return (self.orc.path["tumor_group"] / f"{self.orc.seed}.gz.pkl").is_file()


class Classification(_Problem):
    name = "classification"
    _metrics = ["accuracy", "precision", "recall", "f1", "kappa", "roc_auc"]
    search_best = min

    @staticmethod
    def get_stratification(target):
        """
    The get_stratification function is used to define the stratification of the data.
    The function takes in a target variable and returns a stratified version of that variable.
    In this case, we are simply returning the original target variable.

    Parameters
    ----------
        target
            Define the stratification

    Returns
    -------

        The target variable

    Doc Author
    ----------
        Trelent
    """
        return target

    def evaluate(self):
        """
    The evaluate function takes the results of the models and calculates a number of metrics for each model.
    The metrics are: accuracy, precision, recall, f-score (f-measure), kappa score.


    Parameters
    ----------
        self
            Represent the instance of the class

    Returns
    -------

        A dictionary with the metrics for each model

    Doc Author
    ----------
        Trelent
    """
        y_true = self.orc.datasets["test"][self.orc.target]
        results = self.orc.results

        results["y_true"] = y_true.copy()

        for model in list(models.keys()) + ["ensemble"]:
            if model == "ensemble":
                y_preds = [
                    results[model]["predictions"].round() for model in models.keys()
                ]
                y_pred_round = mode(y_preds).mode.T[:, 0]
                results["ensemble"] = {"predictions": y_pred_round}
            else:
                y_pred_round = results[model]["predictions"]

            metric = {
                "accuracy": balanced_accuracy_score(y_true, y_pred_round),
                "precision": precision_score(y_true, y_pred_round),
                "recall": recall_score(y_true, y_pred_round),
                "f1": f1_score(y_true, y_pred_round),
                "kappa": cohen_kappa_score(y_true, y_pred_round),
                # "roc_auc": roc_auc_score(y_true, y_pred),
            }
            results[model]["metric"] = metric.copy()
        return

    def get_neural_network_params(self):
        """
    The get_neural_network_params function returns a dictionary of the parameters that are used to create
    the neural network. The keys in this dictionary are:

    Parameters
    ----------
        self
            Represent the instance of the class

    Returns
    -------

        The classification_params attribute of the neuralnetwork class

    Doc Author
    ----------
        Trelent
    """
        return NeuralNetwork.classification_params


class Regression(_Problem):
    name = "regression"


problems = {Classification.name: Classification, Regression.name: Regression}
