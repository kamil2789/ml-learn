import numpy as np
import pandas as pd

from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder


def remove_id_column(data: pd.DataFrame) -> pd.DataFrame:
    return data.drop("Id", axis=1)
