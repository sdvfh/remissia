import pandas as pd
from constants import attributes, dataset_types, n_rep, targets, tumors_group
from models import models
from problems import problems
from sklearn.model_selection import train_test_split
from utils import get_path


class Orchestrator:
    def __init__(self):
        """The __init__ function is called when the class is instantiated. It
        sets up the instance variables for this particular object, and does any
        other initialization that might be necessary.

        Parameters
        ----------
            self
                Refer to the current object

        Returns
        -------

            Nothing

        Doc Author
        ----------
            Trelent
        """
        self.path = get_path()
        self.attributes = attributes
        self.target = None
        self.model = None
        self.problem = None
        self.tumor_group = None
        self.seed = None
        self.datasets = {dataset_type: None for dataset_type in dataset_types}
        self.results = {}
        self.list_results = []
        return

    def run(self):
        """The run function is the main function of this module. It runs all
        the models on all the datasets and saves their results in a file.

        Parameters
        ----------
            self
                Access the attributes and methods of the class in python

        Returns
        -------

            Nothing

        Doc Author
        ----------
            Trelent
        """
        for target_name, target_problem in targets:
            self.target = target_name

            problem_class = problems[target_problem]
            self.problem = problem_class(self)

            for tumor_group in tumors_group:
                self.tumor_group = tumor_group
                self.path["tumor_group"] = (
                    self.path["files"] / "ARTICLE" / self.tumor_group
                )
                self.path["tumor_group"].mkdir(parents=True, exist_ok=True)

                for seed in range(n_rep):
                    self.seed = seed
                    if self.problem.checkpoint_exists():
                        continue
                    self._get_dataset()

                    for model_class in models.values():
                        self.model = model_class(self)
                        self.model.run()

                    self.problem.evaluate()
                    self.problem.save_checkpoint()
                    self.results = {}
        return

    def _get_dataset(self):
        """The _get_dataset function is used to load the processed dataset from
        a parquet file. The function also splits the data into train,
        validation and test sets.

        Parameters
        ----------
            self
                Bind the method to an object

        Returns
        -------

            A dictionary of dataframes

        Doc Author
        ----------
            Trelent
        """
        df_path = self.path["datasets"] / "processed" / "processed.snappy.parquet"
        df = pd.read_parquet(df_path)
        (
            self.datasets["train"],
            self.datasets["valid"],
            self.datasets["test"],
        ) = self._get_datasets_uniqueness(df)
        self.datasets["train_valid"] = pd.concat(
            (self.datasets["train"], self.datasets["valid"])
        )
        if self.tumor_group != "ALL":
            for dataset_type in dataset_types:
                dataset = self.datasets[dataset_type]
                self.datasets[dataset_type] = dataset[
                    dataset["TYPE_TUMOR"] == self.tumor_group
                ].copy()
        return

    def _get_datasets_uniqueness(self, df):
        """The _get_datasets_uniqueness function is a helper function that
        takes in the dataframe and returns three datasets: train, valid, test.
        The train_test_split function splits the data into two sets of training
        and testing. The stratify parameter makes a split so that the
        proportion of values in the sample produced will be the same as the
        proportion of values provided to parameter stratify. The
        train_validation_split function splits the training set into two sets:
        validation and training.

        Parameters
        ----------
            self
                Bind the method to an object
            df
                Drop duplicates in the dataframe

        Returns
        -------

            A tuple of 3 dataframes: train, valid and test

        Doc Author
        ----------
            Trelent
        """
        df = df.drop_duplicates(subset=self.attributes)
        train, test = train_test_split(
            df,
            test_size=(1 / 4),
            random_state=self.seed,
            stratify=self.problem.get_stratification(df[self.target]),
        )

        train, valid = train_test_split(
            train,
            test_size=(1 / 3),
            random_state=self.seed,
            stratify=self.problem.get_stratification(train[self.target]),
        )

        train_hash = pd.util.hash_pandas_object(train[self.attributes], index=False)
        valid_hash = pd.util.hash_pandas_object(valid[self.attributes], index=False)
        test_hash = pd.util.hash_pandas_object(test[self.attributes], index=False)

        train = train[~(train_hash.isin(valid_hash) | train_hash.isin(test_hash))]
        valid = valid[~valid_hash.isin(test_hash)]
        return train, valid, test
