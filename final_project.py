import pandas as pd
from sklearn.preprocessing import MinMaxScaler


def scale_01(data:DataFrame)-> DataFrame:
    """Scales all columns of a DataFrame to the range [0, 1] using MinMaxScaler.

    Args:
        data (DataFrame): Input DataFrame.

    Returns:
        DataFrame: Scaled data where each value is in the range [0, 1].
    """
    scale = MinMaxScaler(feature_range=(0,1))
    data = scale.fit_transform(data)
    return pd.DataFrame(data)


def data_processing(data: str = 'data_given.data') -> DataFrame:
    """Loads data, transposes, removes first row, scales remaining rows.

    Args:
        data (str, optional): A file path contaning the data. Defaults to 'data_given.data'.

    Returns:
        DataFrame: processed dataset
    """
    data = pd.read_csv(data, header= None)
    data = data.T
    data = data.drop(index=data.index[0])
    df1 = data.iloc[0]
    df2 = data.iloc[1:]
    df2 = scale_01(df2)
    df1 = pd.DataFrame(data)
    df_final = pd.concat([df1.T, df2])
    return df_final



print(data_processing())


