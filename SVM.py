from typing import Callable

import numpy as np


class SVM:
    """
    Class for the Support Vector Machine model.
    """
    
    def __init__(self, learning_rate: float =0.001, lambda_param: float =0.01, iterations: int = 1000) -> None:
        """
        Constructor.

        Args:
            learning_rate (float) = 0.001: learning rate of the model
            lambda_param (float) = 0.01: regularization parameter
            iterations (int) = 1000: number of iterations for training

        Returns:
            None
        """
        self.learning_rate = learning_rate
        self.lambda_param = lambda_param
        self.iterations = iterations
        self.weights = None
        self.bias = None
        self._gradient = 1

    def fit(self, features: np.ndarray, target: np.ndarray) -> None:
        """
        Converts the input features and target into a trained SVM model.

        Args:
            features (np.ndarray): input features for trainingg
            target (np.ndarray): target values for training

        Returns:
            None
        """
        pass

    def predict(self, features: np.ndarray) -> np.ndarray:
        """
        Predicts the target values for the given input features using the trained SVM model.

        Args:
            features (np.ndarray): input features for prediction

        Returns:
            np.ndarray: predicted target values
        """
        pass
    

    def _hinge_loss(self, features: np.ndarray, target: np.ndarray) -> float:
        """
        Computes the hinge loss for the current model on the given features and target.

        Args:
            features (np.ndarray): input features
            target (np.ndarray): target values

        Returns:
            float: hinge loss value
        """
        # return max(0, 1 - features * target)
        return max(0, 1 - target * (self.weights * features + self.bias))


    def _binary_cross_entropy(self, features: np.ndarray, target: np.ndarray) -> float:
        """
        Computes the binary cross-entropy loss for the current model on the given features and target.

        Args:
            features (np.ndarray): input features
            target (np.ndarray): target values

        Returns:
            float: binary cross-entropy loss value
        """
        return np.mean(sum(features * np.log(target) + (1 - features) * np.log(1 - target)))


    def _compute_gradients(self, features: np.ndarray, target: np.ndarray, loss_func: Callable[...], score: np.ndarray) -> tuple:
        """
        Computes the gradients of the loss function with respect to the model parameters.

        Args:
            features (np.ndarray): input features
            target (np.ndarray): target values
            loss_func (Callable[..., ]): loss function to compute gradients for
            score (np.ndarray): current model scores

        Returns:
            tuple: gradients with respect to weights and bias
        THIS IS COMPLETLEY WRONG
        """
        sample, feature = features.shape

        if target * score >= 1:
            self._gradient = 2 * self.lambda_param * self.weights
        else:
            self._gradient = 2 * self.lambda_param * self.weights - target * feature