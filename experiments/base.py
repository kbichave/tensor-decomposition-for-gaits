"""
Base class for tensor decomposition experiments.

This module provides a shared base class that contains common functionality
for all tensor decomposition experiments, including data loading and tensor
decomposition methods.
"""

import logging
from pathlib import Path
from typing import List, Tuple

import numpy as np
import scipy.io as sio
import tensorly as tl
from tensorly.decomposition import tucker

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TensorExperiment:
    """
    Base class for tensor decomposition experiments on gait data.
    
    This class provides common functionality for loading gait data and
    performing tensor decomposition using Tucker decomposition.
    """

    def __init__(
        self,
        data_dir: str = "Data",
        num_test: int = 10,
        num_train: int = 20,
    ):
        """
        Initialize the tensor experiment with gait data.
        
        Args:
            data_dir: Directory containing the .mat data files
            num_test: Number of test samples per class
            num_train: Number of training samples per class
        """
        self.data_dir = Path(data_dir)
        self.num_test = num_test
        self.num_train = num_train
        self.acc_dict: List = []

        # Load gait data tensors
        self._load_data()

    def _load_data(self) -> None:
        """Load gait data from .mat files."""
        try:
            recovered10_path = self.data_dir / "recovered10.mat"
            recovered50_path = self.data_dir / "recovered50.mat"
            recovered95_path = self.data_dir / "recovered95.mat"

            recovered10 = sio.loadmat(str(recovered10_path))
            self.recovered10 = recovered10["recovered10"]

            recovered50 = sio.loadmat(str(recovered50_path))
            self.recovered50 = recovered50["recovered50"]

            recovered95 = sio.loadmat(str(recovered95_path))
            self.recovered95 = recovered95["recovered95_1"]

            logger.info(
                f"Loaded data: recovered10 shape={self.recovered10.shape}, "
                f"recovered50 shape={self.recovered50.shape}, "
                f"recovered95 shape={self.recovered95.shape}"
            )
        except FileNotFoundError as e:
            logger.error(f"Data file not found: {e}")
            raise
        except KeyError as e:
            logger.error(f"Expected key not found in .mat file: {e}")
            raise

    def decomposed(
        self, factor: List[np.ndarray], recovered: np.ndarray, l: int
    ) -> np.ndarray:
        """
        Decompose a tensor based on factors obtained from Tucker decomposition.
        
        This method applies the Tucker decomposition factors to reduce the
        dimensionality of the input tensor.
        
        Args:
            factor: List of factor matrices obtained from Tucker decomposition.
                   factor[1] corresponds to mode-1, factor[2] to mode-2.
            recovered: Input tensor of shape (num_samples, 3, 283)
            l: Reduced dimension (target dimensionality)
        
        Returns:
            Decomposed matrix of shape (num_samples, l)
        """
        # Unfold tensor along mode-1 and apply factor[1]
        unfolded = np.dot(np.transpose(factor[1]), tl.unfold(recovered, mode=1))
        
        # Fold back and apply factor[2]
        decompose = tl.fold(unfolded, mode=1, shape=[recovered.shape[0], 1, 283])
        decompose = np.dot(decompose, factor[2])
        
        # Reshape to final form
        return np.reshape(decompose, (recovered.shape[0], l))

    def run_exp(self) -> None:
        """
        Run the experiment.
        
        This method should be implemented by subclasses to define
        the specific experiment logic.
        """
        raise NotImplementedError("Subclasses must implement run_exp()")

