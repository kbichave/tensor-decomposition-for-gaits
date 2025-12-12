"""
Accuracy vs Reduced Dimension vs Samples experiment.

This experiment evaluates classification accuracy as a function of both
the reduced dimension parameter and the number of training samples.
"""

import logging
import random
from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier, NearestCentroid
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from tensorly.decomposition import tucker

from experiments.base import TensorExperiment

logger = logging.getLogger(__name__)


class AccVsReducedDimensionVsSamples(TensorExperiment):
    """
    Experiment to evaluate accuracy vs both reduced dimension and samples.
    
    This class performs a 2D sweep over reduced dimension and number of
    training samples, evaluating classification accuracy using kNN, MLP,
    SVM, and Nearest Centroid classifiers.
    """

    def __init__(self, data_dir: str = "Data", num_test: int = 4, num_train: int = 20):
        """
        Initialize the accuracy vs reduced dimension vs samples experiment.
        
        Args:
            data_dir: Directory containing the .mat data files
            num_test: Number of test samples per class
            num_train: Number of training samples per class (not used in this experiment)
        """
        super().__init__(data_dir=data_dir, num_test=num_test, num_train=num_train)
        
        # Split test data from training data (fixed split for this experiment)
        self.recovered10_test = self.recovered10[-self.num_test :, :, :]
        self.recovered10 = self.recovered10[: len(self.recovered10) - 4, :, :]

        self.recovered50_test = self.recovered50[-self.num_test :, :, :]
        self.recovered50 = self.recovered50[: len(self.recovered50) - 4, :, :]

        self.recovered95_test = self.recovered95[-self.num_test :, :, :]
        self.recovered95 = self.recovered95[: len(self.recovered95) - 4, :]

    def run_exp(self) -> None:
        """
        Run the accuracy vs reduced dimension vs samples experiment.
        
        Performs a 2D sweep over reduced dimension (2 to 283, step 10) and
        number of samples (2 to 29), evaluating classification accuracy over
        500 random realizations.
        """
        acc_dict: List[List[np.ndarray]] = []

        # Loop over reduced dimension
        for l in range(2, 283, 10):
            acc_samples: List[np.ndarray] = []
            
            # Loop over number of samples
            for samples in range(2, 30):
                acc: List[np.ndarray] = []
                
                # Repeat experiment multiple times for statistical significance
                for repeat in range(1, 500):
                    # Randomly sample training data
                    _recovered10 = self.recovered10[
                        random.sample(range(len(self.recovered10)), samples), :, :
                    ]
                    _recovered50 = self.recovered50[
                        random.sample(range(len(self.recovered50)), samples), :, :
                    ]
                    _recovered95 = self.recovered95[
                        random.sample(range(len(self.recovered95)), samples), :, :
                    ]

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

                    logger.info(f"Reduced Dimension: {l}, Sample: {samples}, Repeat: {repeat}")

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
                        solver="lbfgs", alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1
                    )
                    clf.fit(xtrain, ytrain)
                    ypreds_mlp = clf.predict(xtest)

                    # Train and evaluate SVM classifier
                    clf = SVC(gamma="auto")
                    clf.fit(xtrain, ytrain)
                    ypreds_svm = clf.predict(xtest)

                    # Train and evaluate Nearest Centroid classifier
                    clf = NearestCentroid()
                    clf.fit(xtrain, ytrain)
                    ypreds_nc = clf.predict(xtest)

                    # Store accuracies
                    acc.append(
                        np.array(
                            [
                                accuracy_score(ytest, ypreds_knn),
                                accuracy_score(ytest, ypreds_mlp),
                                accuracy_score(ytest, ypreds_svm),
                                accuracy_score(ytest, ypreds_nc),
                            ]
                        )
                    )

                acc_samples.append(np.mean(acc, axis=0))
            acc_dict.append(acc_samples)

        # Save accuracies as checkpoint
        output_path = Path("acc_dict.txt")
        with open(output_path, "w") as f:
            for item in np.array(acc_dict):
                f.write("%s\n" % item)

        acc_dict = np.array(acc_dict)

        # Plotting - 3D surface plot
        _classifiers = ["kNN", "MLP", "SVM", "Nearest Centroid"]
        hf = plt.figure()
        ha = hf.add_subplot(111, projection="3d")
        x = [i for i in range(2, 283, 10)]
        y = [i for i in range(2, 30)]
        X, Y = np.meshgrid(x, y)  # `plot_surface` expects `x` and `y` data to be 2D
        for _ in range(4):
            ha.plot_surface(X, Y, acc_dict[:, :, _], label=_classifiers[_])
        plt.show()
