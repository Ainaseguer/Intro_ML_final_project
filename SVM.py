import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.decomposition import PCA


class SVM(BaseEstimator, ClassifierMixin):
    """
    Class for the Support Vector Machine model.
    """

    def __init__(
        self,
        learning_rate: float = 0.001,
        c_param: float = 0.01,
        iterations: int = 1000,
    ) -> None:
        """
        Constructor that initialises the learning rate, the c parameter
        and the iterations

        Args:
            learning_rate (float) = 0.001: learning rate of the model
            c_param (float) = 0.01: regularization parameter
            iterations (int) = 1000: number of iterations for training

        Returns:
            None
        """
        self.learning_rate = learning_rate
        self.c_param = c_param
        self.iterations = iterations
        self.weights = None
        self.bias = None

    def get_params(self, deep=True) -> dict:
        """
        Get parameters for sklearn compatibility.

        Returns:
            dict: parameters of the model
        """
        return {
            "learning_rate": self.learning_rate,
            "c_param": self.c_param,
            "iterations": self.iterations,
        }

    def set_params(self, **params) -> "SVM":
        """
        Set parameters for sklearn compatibility.

        Args:
            **params: parameters to set
        Returns:
            self
        """
        for key, value in params.items():
            setattr(self, key, value)

        return self

    def fit(self, features: np.ndarray, target: np.ndarray) -> None:
        """
        Trains the SVM model on the given features and target values.

        Args:
            features (np.ndarray): input features for trainingg
            target (np.ndarray): target values for training

        Returns:
            None
        """
        X = np.array(features)
        y = np.array(target)

        # This line is for Sklearn compatibility
        self.classes_ = np.unique(y)

        n_samples, n_features = features.shape

        # Initializing weights and bias
        if self.weights is None:
            self.weights = np.zeros(n_features)
            self.bias = 0

        # Converting target labels to -1 and 1 for SVM
        y_srv = np.where(y <= 0, -1, 1)

        for _ in range(self.iterations):
            # Iterating through each data point
            for idx, x_i in enumerate(X):
                # Checking the condition y_i * (w * x_i + b) >= 1
                condition = y_srv[idx] * (np.dot(x_i, self.weights) + self.bias) >= 1
                # True condition
                if condition:
                    # Updating weights based on regularization
                    self.weights -= self.learning_rate * (
                        2 * self.c_param * self.weights
                    )
                else:
                    # Updating weights based on regularization and misclassification
                    gradient = 2 * self.c_param * self.weights - (y_srv[idx] * x_i)
                    self.weights -= self.learning_rate * gradient

                    # Updating bias = -y_i
                    self.bias -= self.learning_rate * (-y_srv[idx])

    def predict(self, features: np.ndarray) -> np.ndarray:
        """
        Predicts the target values for the given input features using the trained SVM model.

        Args:
            features (np.ndarray): input features for prediction

        Returns:
            np.ndarray: predicted target values
        """
        X = np.asarray(features)

        # Calculating the hyperplane output = w * x + b
        hyperplane_output = np.dot(X, self.weights) + self.bias
        predictions = np.sign(hyperplane_output)

        # Returning the labels 0 and 1
        return np.where(predictions == -1, 0, 1)

    def labels(self, predictions: np.ndarray) -> np.ndarray:
        """
        Returns string labels 'B' and 'M'. 'B' for 0 and 'M' for 1.

        Args:
            features (np.ndarray): predictions as 0 and 1

        Returns:
            np.ndarray: predicted target labels as 'B' and 'M'
        """
        return np.where(predictions == 0, "B", "M")

    def _hinge_loss(self, features: np.ndarray, target: np.ndarray) -> float:
        """
        Computes the average Hinge loss for the current model
        on the given features and target.

        Formula: mean(max(0, 1 - y_i * (w * x_i + b)))

        Args:
            features (np.ndarray): input features
            target (np.ndarray): target values

        Returns:
            float: average Hinge loss value
        """
        # y must be -1 or 1
        y_srv = np.where(target <= 0, -1, 1)

        # Calculating raw scores
        scores = np.dot(features, self.weights) + self.bias

        # Calculating Hinge loss
        losses = np.maximum(0, 1 - y_srv * scores)

        # Returning average Hinge loss value
        return np.mean(losses)


