import pandas as pd
import numpy as np
from os import path, makedirs


def read(filepath, encoding='ISO-8859-1', sep=",", dtype=str):
    return pd.read_csv(filepath, encoding=encoding, sep=sep, dtype=dtype)



