import pandas as pd
from sklearn.preprocessing import MinMaxScaler


def scale_01(data: pd.DataFrame )-> pd.DataFrame :
    """Scales all columns of a DataFrame to the range [0, 1] using MinMaxScaler.

    Args:
        data (DataFrame): Input DataFrame.

    Returns:
        DataFrame: Scaled data where each value is in the range [0, 1].
    """
    scale = MinMaxScaler(feature_range=(0,1))
    data = scale.fit_transform(data)
    return pd.DataFrame(data)

def letter_to_number(data: pd.Series) -> pd.Series:
    for i in range(len(data)):
        if data.loc[i] == "B":
            data.loc[i] = 0
        elif data.loc[i] == "M":
            data.loc[i] = 1
        else:
            raise ValueError("Unexpected label")
    return data


def data_processing(data: str = 'data_given.data') -> pd.DataFrame:
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
    df1 = letter_to_number(df1)

    df2 = data.iloc[1:]
    df2 = scale_01(df2)

    df1 = pd.DataFrame(df1).T
    df_final = pd.concat([df1, df2], axis=0, ignore_index=True)
    return df_final



print(data_processing())


