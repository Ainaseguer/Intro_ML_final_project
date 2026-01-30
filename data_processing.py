import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler


def extracting_features_and_target(
    data_path: str = "data_given.data",
) -> tuple[np.ndarray, np.ndarray]:
    """
    Separates features and target from the dataset, before any processing.

    Args:
        data (pd.DataFrame): The input DataFrame containing features and target.

    Returns:
        tuple[np.ndarray, np.ndarray]: A tuple containing the features array (X)
        and the target array (y).
    """
    data = pd.read_csv(filepath_or_buffer=data_path, header=None)

    # Seperating the Target (y) and encoding the labels
    y = data.iloc[:, 1]
    y = letter_to_number(y)

    # Extracting Features (X)
    X = data.iloc[:, 2:]

    return X.to_numpy(), y.to_numpy()


def letter_to_number(data: pd.Series) -> pd.Series:
    """
    Encodes the labels from the dataset from 'B' and 'M'
    to be 0 and 1.

    Args:
        data (pd.Series): The Target Series containing 'B' and 'M' labels.

    Returns:
        pd.Series: processed Target column with 0 and 1 labels
    """
    # Mapping the labels
    data = data.map({"B": 0, "M": 1})

    # Check for unexpected values
    if any(data == ""):
        raise ValueError("Unexpected label")

    return data


def preprocessing_pipeline(n_components: int | float = 0.95) -> Pipeline:
    """
    Creates a preprocessing pipeline that scales features and applies PCA.

    Args:
        n_components: Number of principal components for PCA.

    Returns:
        Pipeline: A sklearn Pipeline object for preprocessing.
    """
    return Pipeline(
        [
            ("scaler", MinMaxScaler(feature_range=(0, 1))),
            ("pca", PCA(n_components=n_components, random_state=42)),
        ]
    )



def preprocess_train_test(
    X_train: np.ndarray, X_test: np.ndarray, n_components: int | float = 0.95
) -> tuple[np.ndarray, np.ndarray]:
    """
    Preprocesses training and test feature sets using normalization and PCA.

    Args:
        X_train: Training feature matrix.
        X_test: Test feature matrix.
        n_components: Number of principal components for PCA.

    Returns:
        tuple[np.ndarray, np.ndarray]: Preprocessed training and test feature matrices.
    """
    # Create preprocessing pipeline
    pipeline = preprocessing_pipeline(n_components=n_components)

    # Fit and transform training data
    X_train_processed = pipeline.fit_transform(X_train)

    # Transform test data using the same pipeline
    X_test_processed = pipeline.transform(X_test)

    return X_train_processed, X_test_processed
