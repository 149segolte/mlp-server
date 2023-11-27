import sys
import json
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder, OneHotEncoder, StandardScaler, MinMaxScaler, MaxAbsScaler, RobustScaler, QuantileTransformer
from io import StringIO


def handle_empty(df, empty):
    if empty == 'drop':
        df = df.dropna()
    elif empty == 'mean':
        df = df.fillna(df.mean())
    elif empty == 'median':
        df = df.fillna(df.median())
    elif empty == 'mode':
        df = df.fillna(df.mode().iloc[0])
    return df


def handle_categorical(df, categorical, target):
    df_target = df[target]
    df = df.drop(columns=[target])
    if categorical == 'onehot':
        df = pd.get_dummies(df, columns=df.select_dtypes(
            include=['object']).columns)
    elif categorical == 'label':
        df = pd.DataFrame(LabelEncoder().fit_transform(
            df), columns=df.columns)
    elif categorical == 'ordinal':
        df = pd.DataFrame(OrdinalEncoder().fit_transform(
            df), columns=df.columns)
    elif categorical == 'binary':
        df = pd.DataFrame(LabelEncoder().fit_transform(
            df), columns=df.columns)
        df = pd.get_dummies(df, columns=df.select_dtypes(
            include=['object']).columns)
    return df.join(df_target)


def handle_scaling(df, scale):
    if scale == 'Standard':
        df = pd.DataFrame(StandardScaler().fit_transform(df),
                          columns=df.columns)
    elif scale == 'MinMax':
        df = pd.DataFrame(MinMaxScaler().fit_transform(df), columns=df.columns)
    elif scale == 'MaxAbs':
        df = pd.DataFrame(MaxAbsScaler().fit_transform(df), columns=df.columns)
    elif scale == 'Robust':
        df = pd.DataFrame(RobustScaler().fit_transform(df), columns=df.columns)
    elif scale == 'Quantile':
        df = pd.DataFrame(QuantileTransformer(
        ).fit_transform(df), columns=df.columns)
    return df


if __name__ == '__main__':
    input_file = sys.argv[1]
    f = open(input_file, 'r')
    data = json.load(f)
    f.close()
    '''
        structure of data is:
        {
            "name": "mesa",
            "target": "target_col_name",
            "empty": "drop" | "mean" | "median" | "mode",
            "categorical": "onehot" | "label" | "ordinal" | "binary",
            "scale": "Standard" | "MinMax" | "MaxAbs" | "Robust" | "Quantile",
            "file": {
                "name": "file_name",
                "type": "text/csv",
                "content": plain_text csv,
            },
        }
    '''

    # load data
    df = pd.read_csv(StringIO(data['file']['content']))
    print(df.info())

    empty = data['empty']
    categorical = data['categorical']
    scale = data['scale']
    target = data['target']

    print(df.info())
    # check if empty values
    if df.isnull().values.any():
        df = handle_empty(df, empty)

    print(df.info())
    # check if categorical
    if df.select_dtypes(include=['object']).shape[1] > 0:
        df = handle_categorical(df, categorical, target)

    print(df.info())
    # check if scaling
    df = handle_scaling(df, scale)

    print(df.info())
    # encode target if categorical
    if df[target].dtype == 'object':
        df[target] = LabelEncoder().fit_transform(df[target])

    df = df.astype(float)
    print(df.head())
    # save data
    str1 = df.to_csv(index=False)
    data['file']['content'] = str1
    f = open(input_file, 'w')
    json.dump(data['file'], f)
    f.close()
