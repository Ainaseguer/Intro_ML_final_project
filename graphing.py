import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split, validation_curve
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler

from data_processing import (
    extracting_features_and_target,
    preprocessing_pipeline,
)
from SVM import SVM


def plot_validation_curve() -> None:
    """
    Plots the validation curve for the SVM model by varying the regularization
    parameter C and evaluating training and cross-validation accuracy.

    Args:
        None

    Returns:
        None
    """
    X_raw, y = extracting_features_and_target()

    pipeline = Pipeline(
        [
            ("preprocessing", preprocessing_pipeline(n_components=0.95)),
            ("svm", SVM()),
        ]
    )

    # Define the range of C values to test
    param_range = [0.0001, 0.001, 0.01, 0.1, 1, 10]

    # Compute validation curve
    train_scores, test_scores = validation_curve(
        pipeline,
        X_raw,
        y,
        param_name="svm__c_param",
        param_range=param_range,
        cv=5,
        scoring="accuracy",
        n_jobs=-1,
    )

    # Calculate mean scores
    train_mean = np.mean(train_scores, axis=1)
    test_mean = np.mean(test_scores, axis=1)

    # Plot validation curve
    plt.figure(figsize=(10, 6))
    plt.semilogx(
        param_range, train_mean, label="Training Accuracy", color="blue", marker="o"
    )
    plt.semilogx(
        param_range,
        test_mean,
        label="Cross-Validation Accuracy",
        color="red",
        marker="o",
    )
    plt.title("Validation Curve: Relationship between C and the Model's Accuracy")
    plt.xlabel("C (Regularization Strength)")
    plt.ylabel("Accuracy Score")
    plt.legend(loc="best")
    plt.grid(True)
    plt.show()


def plot_loss_curve(iterations: int = 1000, learning_rate: float = 0.001) -> None:
    """
    Plots the hinge loss curve for the SVM model over a specified number of iterations.

    Args:
        iterations (int): Number of iterations to train the model.
        learning_rate (float): Learning rate for the SVM model.

    Returns:
        None
    """
    X_raw, y = extracting_features_and_target()

    # Split the data into training and testing sets
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X_raw, y, test_size=0.2, random_state=42, stratify=y
    )

    # Preprocess the data
    preprocessing = preprocessing_pipeline(n_components=0.95)
    X_train = preprocessing.fit_transform(X_train_raw)
    X_test = preprocessing.transform(X_test_raw)

    # Initialize SVM model with specified learning rate
    model = SVM(iterations=1, learning_rate=learning_rate)
    model.weights = np.zeros(X_train.shape[1])
    model.bias = 0

    # Train the model and record hinge loss at each iteration
    train_loss_history = []
    test_loss_history = []
    for i in range(iterations):
        model.fit(X_train, y_train)

        # Track the hinge loss on the training set
        current_train_loss = model._hinge_loss(X_train, y_train)
        train_loss_history.append(current_train_loss)

        # Track the hinge loss on the test set
        current_test_loss = model._hinge_loss(X_test, y_test)
        test_loss_history.append(current_test_loss)

    # Plot the training hinge loss curve
    plt.figure(figsize=(10, 6))
    plt.plot(range(iterations), train_loss_history, color="green", linewidth=2)
    plt.xlabel("Iterations")
    plt.ylabel("Average Hinge Loss on the training set")
    plt.title("Graph showing Hinge Loss of SVM model on the training set")
    plt.grid(True, alpha=0.3)
    plt.show()

    # Plot the test hinge loss curve
    plt.figure(figsize=(10, 6))
    plt.plot(range(iterations), test_loss_history, color="green", linewidth=2)
    plt.xlabel("Iterations")
    plt.ylabel("Average Hinge Loss on the test set")
    plt.title("Graph showing Hinge Loss of SVM model on the test set")
    plt.grid(True, alpha=0.3)
    plt.show()


def plot_pca(n_components: int = 2) -> None:
    """
    Plots the PCA results of the dataset.

    Args:
        n_components (int): Number of principal components to reduce to.
            Default is 2 for 2D plotting.

    Returns:
        None
    """
    # Preprocess the data and apply PCA
    processing_pipeline = preprocessing_pipeline(n_components=n_components)
    X_raw, y = extracting_features_and_target()
    X_pca = processing_pipeline.fit_transform(X_raw)

    # Plot the PCA results
    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap="viridis", alpha=0.7)
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.title("Graph showing dimensional reduction using PCA of Breast Cancer Dataset")
    plt.legend(handles=scatter.legend_elements()[0], labels=["Benign", "Malignant"])
    plt.show()


def plot_elbow_method() -> None:
    """
    Plots the elbow method for determining the optimal number of components in PCA.

    Args:
        None

    Returns:
        None
    """
    X_raw, y = extracting_features_and_target()

    # Splitting the data
    X_train_raw, _, _, _ = train_test_split(
        X_raw, y, test_size=0.2, random_state=42, stratify=y
    )

    # Scale the features to [0, 1] range
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train_raw)

    # Apply PCA to the dataset and compute explained variance
    pca = PCA(n_components=min(X_train_scaled.shape))
    pca.fit(X_train_scaled)
    cumulative_variance = np.cumsum(pca.explained_variance_ratio_)

    # Plot the elbow method graph
    plt.figure(figsize=(10, 6))
    plt.plot(
        range(1, len(cumulative_variance) + 1),
        cumulative_variance,
        marker="o",
        color="b",
    )
    plt.axhline(y=0.95, color="r", linestyle="--", label="95% Explained Variance")
    plt.title(
        "Relationship between Number of Principal Components and Cumulative Explained Variance"
    )
    plt.xlabel("Number of Principal Components")
    plt.ylabel("Cumulative Explained Variance")
    plt.grid(True)
    plt.legend()
    plt.show()


# Plotting all the graphs
if __name__ == "__main__":
    print("Generating validation curve...")
    plot_validation_curve()

    print("\nGenerating loss curve...")
    plot_loss_curve()

    print("\nGenerating elbow method plot...")
    plot_elbow_method()

    print("\nGenerating PCA plot...")
    plot_pca()
