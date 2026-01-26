import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler


def normalization(data: np.ndarray) -> pd.DataFrame:
    """
    Normalizes the data by scaling all columns of a DataFrame to the range[0, 1]
    using MinMaxScaler.

    Args:
        data (np.ndarray): Input features array.

    Returns:
        DataFrame: Scaled data where each value is in the range [0, 1].
    """
    # Scaling the data to range [0,1]
    scaler = MinMaxScaler(feature_range=(0, 1))
    data = scaler.fit_transform(data)

    return pd.DataFrame(data)


def letter_to_number(data: pd.Series) -> pd.Series:
    """
    Encodes the labels from the dataset from 'M' and 'B'
    to be 0 and 1.

    Args:
        data (pd.Series): The Target Series containing 'M' and 'B' labels.

    Returns:
        pd.Series: processed Target column with 0 and 1 labels
    """
    # Mapping the labels
    data = data.map({"B": 0, "M": 1})

    # Check for unexpected values
    if any(data == ""):
        raise ValueError("Unexpected label")

    return data


def data_processing(
    data_path: str = "data_given.data",
) -> tuple[np.ndarray, np.ndarray]:
    """
    Loads data, transposes, removes first row, and scales remaining rows.

    Args:
        data_path (str, optional): A file path containing the data.
        Defaults to 'data_given.data'.

    Returns:
        tuple[pd.DataFrame, pd.Series]: A tuple containing the processed
        features DataFrame (X) and Target Series (y).
    """
    # Loading the data
    data = pd.read_csv(filepath_or_buffer=data_path, header=None)
    # Transposition
    # data = data.T
    # Removing the first row
    data = data.drop(index=data.index[0])

    # Seperating the Target (y) and encoding the labels
    # y = data.iloc[0]
    # y = letter_to_number(y)
    y = letter_to_number(data.iloc[:, 1])

    # Scaling the Features (X) to be in range [0, 1]
    # X = data.iloc[1:].T
    # X = normalization(X.to_numpy())
    X_raw = data.iloc[:, 2:].to_numpy()
    X = normalization(X_raw)

    return X.to_numpy(), y.to_numpy()
