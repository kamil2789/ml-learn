import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder


def one_hot_encode_data(data: pd.DataFrame, variables: list) -> pd.DataFrame:
    onehot = OneHotEncoder(sparse_output=False)
    encoded_data = pd.DataFrame(onehot.fit_transform(data[variables]))
    encoded_data.columns = onehot.get_feature_names_out(variables)
    data = pd.concat([data, encoded_data], axis=1)
    data.drop(labels=variables, axis="columns", inplace=True)
    return data
