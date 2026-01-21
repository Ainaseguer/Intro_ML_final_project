import numpy as np


class SVM:
    """
    Class for the Support Vector Machine model.
    """

    def __init__(
        self,
        learning_rate: float = 0.001,
        lambda_param: float = 0.01,
        iterations: int = 1000,
        weights: np.ndarray = np.array([]),
        bias: float = 0.0,
    ) -> None:
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
        self.weights = weights
        self.bias = bias

    def fit(self, features: np.ndarray, target: np.ndarray) -> None:
        """
        Trains the SVM model on the given features and target values.

        Args:
            features (np.ndarray): input features for trainingg
            target (np.ndarray): target values for training

        Returns:
            None
        """
        n_samples, n_features = features.shape

        # initializing weights and bias
        self.weights = np.zeros(n_features)
        self.bias = 0

        # Converting target labels to -1 and 1 for SVM
        y_srv = np.where(target <= 0, -1, 1)

        for _ in range(self.iterations):
            # Iterating through each data point
            for idx, x_i in enumerate(features):
                # Checking the condition y_i * (w * x_i + b) >= 1
                condition = y_srv[idx] * (np.dot(x_i, self.weights) + self.bias) >= 1
                # True condition
                if condition:
                    # Updating weights based on regularization
                    self.weights -= self.learning_rate * (
                        2 * self.lambda_param * self.weights
                    )
                else:
                    # Updating weights based on regularization and misclassification
                    gradient = 2 * self.lambda_param * self.weights - (y_srv[idx] * x_i)
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
        # Calculating the hyperplane output = w * x + b
        hyperplane_output = np.dot(features, self.weights) + self.bias
        predictions = np.sign(hyperplane_output)

        # Returning the original labels 'B' and 'M'
        return np.where(predictions == -1, "B", "M")

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

    def _binary_cross_entropy(self, features: np.ndarray, target: np.ndarray) -> float:
        """
        Computes the binary cross-entropy loss for the current model
        on the given features and target.

        Args:
            features (np.ndarray): input features
            target (np.ndarray): target values

        Returns:
            float: binary cross-entropy loss value
        """
        # Converting Targets to probabilities
        y_prob = np.where(target <= 0, 0, 1)

        # Calculating raw scores
        scores = np.dot(features, self.weights) + self.bias

        # Clipping the scores to prevent overfloe in exponents
        scores = np.clip(scores, -500, 500)
        # Applying sigmoid to calculate probabilities
        probabilities = 1 / (1 + np.exp(-scores))

        # Adjusting probabilities to avoid log(0)
        epsilon = 1e-15
        probabilities = np.clip(probabilities, epsilon, 1 - epsilon)

        # Calculating BCE
        loss = -np.mean(y_prob * np.log(probabilities) + (1 - y_prob) * np.log(1 - probabilities))


        # Returning the loss
        return loss

    def calculate_loss(
        self,
        features: np.ndarray,
        target: np.ndarray,
        loss_func: str,
    ) -> float:
        """
        Computes the gradients of the loss function with respect to the model parameters.

        Args:
            features (np.ndarray): input features
            target (np.ndarray): target values
            loss_func (Callable[..., ]): loss function to compute gradients for

        Returns:
            tuple: gradients with respect to weights and bias
        THIS IS COMPLETLEY WRONG
        """
        if loss_func == "hinge":
            return self._hinge_loss(features, target)
        elif loss_func == "bce":
            return self._binary_cross_entropy(features, target)
        else:
            raise ValueError(f"Unknown loss function: {loss_func}")
