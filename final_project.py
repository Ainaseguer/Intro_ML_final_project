import pandas as pd
from sklearn.preprocessing import MinMaxScaler


def scale_01(data):
    scale = MinMaxScaler(feature_range=(0,1))
    data = scale.fit_transform(data)
    return pd.DataFrame(data)


def data_processing(data= 'data_given.data'):
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


