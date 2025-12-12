"""
Accuracy vs Samples experiment.

This experiment evaluates classification accuracy as a function of
the number of training samples per class.
"""

import logging
import random
from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from tensorly.decomposition import tucker

from experiments.base import TensorExperiment

logger = logging.getLogger(__name__)


class AccVsSamples(TensorExperiment):
    """
    Experiment to evaluate accuracy vs number of training samples.
    
    This class performs Tucker decomposition with a fixed reduced dimension
    and evaluates classification accuracy as the number of training samples
    varies, using kNN, MLP, and SVM classifiers.
    """

    def __init__(self, data_dir: str = "Data", num_test: int = 10, num_train: int = 20):
        """
        Initialize the accuracy vs samples experiment.
        
        Args:
            data_dir: Directory containing the .mat data files
            num_test: Number of test samples per class
            num_train: Number of training samples per class (not used in this experiment)
        """
        super().__init__(data_dir=data_dir, num_test=num_test, num_train=num_train)

    def run_exp(self) -> None:
        """
        Run the accuracy vs samples experiment.
        
        Varies the number of training samples from 2 to 23 and evaluates
        classification accuracy over 600 random realizations with a fixed
        reduced dimension of 11.
        """
        acc_dict: List[np.ndarray] = []
        l = 11  # Fixed reduced dimension

        # Loop over number of samples
        for samples in range(2, 24):
            acc: List[np.ndarray] = []
            
            # Repeat experiment multiple times for statistical significance
            for repeat in range(1, 600):
                # Randomly sample training and test data
                _recovered10 = self.recovered10[
                    random.sample(range(len(self.recovered10)), samples + self.num_test),
                    :,
                    :,
                ]
                _recovered50 = self.recovered50[
                    random.sample(range(len(self.recovered50)), samples + self.num_test),
                    :,
                    :,
                ]
                _recovered95 = self.recovered95[
                    random.sample(range(len(self.recovered95)), samples + self.num_test),
                    :,
                    :,
                ]

                # Split into test and training sets
                self.recovered10_test = _recovered10[: self.num_test, :, :]
                self.recovered50_test = _recovered50[: self.num_test, :, :]
                self.recovered95_test = _recovered95[: self.num_test, :, :]

                _recovered10 = _recovered10[self.num_test :, :, :]
                _recovered50 = _recovered50[self.num_test :, :, :]
                _recovered95 = _recovered95[self.num_test :, :, :]

                # Apply Tucker decomposition to obtain core and factors
                result10 = tucker(_recovered10, rank=[_recovered10.shape[0], 1, l])
                factor10 = result10.factors
                result50 = tucker(_recovered50, rank=[_recovered50.shape[0], 1, l])
                factor50 = result50.factors
                result95 = tucker(_recovered95, rank=[_recovered95.shape[0], 1, l])
                factor95 = result95.factors

                # Decompose tensors using obtained factors
                _decomposed10 = self.decomposed(factor10, _recovered10, l)
                _decomposed50 = self.decomposed(factor50, _recovered50, l)
                _decomposed95 = self.decomposed(factor95, _recovered95, l)

                test_decomposed10 = self.decomposed(factor10, self.recovered10_test, l)
                test_decomposed50 = self.decomposed(factor50, self.recovered50_test, l)
                test_decomposed95 = self.decomposed(factor95, self.recovered95_test, l)

                logger.info(f"Sample: {samples}, Repeat: {repeat}")

                # Prepare training data
                _Y = np.ravel(np.array([[1] * samples + [2] * samples + [3] * samples]))
                X = np.concatenate((_decomposed10, _decomposed50, _decomposed95))

                # Prepare test data
                xtrain = X
                xtest = np.concatenate((test_decomposed10, test_decomposed50, test_decomposed95))
                ytrain = _Y
                ytest = np.ravel(
                    np.array([[1] * self.num_test + [2] * self.num_test + [3] * self.num_test])
                )

                # Train and evaluate kNN classifier
                clf = KNeighborsClassifier(n_neighbors=3)
                clf.fit(xtrain, ytrain)
                ypreds_knn = clf.predict(xtest)

                # Train and evaluate MLP classifier
                clf = MLPClassifier(
                    solver="lbfgs",
                    activation="relu",
                    alpha=1e-5,
                    early_stopping=True,
                    hidden_layer_sizes=(16, 32, 64, 32, 16, 2),
                    random_state=1,
                    max_iter=5000,
                )
                clf.fit(xtrain, ytrain)
                ypreds_mlp = clf.predict(xtest)

                # Train and evaluate SVM classifier
                clf = SVC(gamma="auto")
                clf.fit(xtrain, ytrain)
                ypreds_svm = clf.predict(xtest)

                # Store accuracies
                acc.append(
                    np.array(
                        [
                            accuracy_score(ytest, ypreds_knn),
                            accuracy_score(ytest, ypreds_mlp),
                            accuracy_score(ytest, ypreds_svm),
                        ]
                    )
                )

            acc_dict.append(np.mean(acc, axis=0))

        # Save accuracies as checkpoint
        output_path = Path("acc_dict.txt")
        with open(output_path, "w") as f:
            for item in np.array(acc_dict):
                f.write("%s\n" % item)

        acc_dict = np.array(acc_dict)

        # Plotting
        _classifiers = ["kNN", "MLP", "SVM"]
        fig = plt.figure()
        for _ in range(3):
            plt.plot([i for i in range(2, 24)], acc_dict[:, _], label=_classifiers[_])
        fig.suptitle("Accuracy vs Samples")
        plt.xlabel("Samples")
        plt.ylabel("Accuracy")
        plt.legend(loc="lower right")
        
        figures_dir = Path("Figures")
        figures_dir.mkdir(exist_ok=True)
        plt.savefig(figures_dir / "acc_vs_samples.png")
        plt.show()
