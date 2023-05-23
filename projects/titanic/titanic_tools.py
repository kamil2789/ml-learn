import pandas as pd
import numpy as np


def load_dataset(path: str) -> pd.DataFrame:
    pd.read_csv(path)
