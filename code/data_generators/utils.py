# # Copyright (C) KonaAI - All Rights Reserved
"""This module loads all underlying constants"""
import configparser
import os

import numpy as np

config_file_path = os.path.join(os.path.dirname(__file__), "config.ini")
config = configparser.ConfigParser()
config.optionxform = str
config.read(config_file_path, encoding="utf-8")


def set_random_to_null(df, fraction=0.1, include_columns=[]):
    """Set random values to NaN in a given fraction of rows"""
    for col in include_columns:
        df[col] = df[col].mask(
            np.random.choice(
                [True, False], size=df[col].shape, p=[fraction, 1 - fraction]
            )
        )
    return df
