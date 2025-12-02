# # Copyright (C) KonaAI - All Rights Reserved
"""This module provides scaling functions"""


def scale_zero_one(_s):
    """Scale a series between 0 and 1 for a given series data

    Args:
    ----
        _s (pd.Series): Series to scale

    Returns:
        pd.Series: Scaled series
    """
    return (_s - _s.min()) / (_s.max() - _s.min()) if (_s.max() - _s.min()) > 0 else _s


def scale_zero_one_with_min_max(_s, _min, _max):
    """This function scales a series between 0 and 1 for a given series data and given min and max values

    Args:
    ----
        _s (pd.Series): Input data series
        _min (int): Mininum value to benchmark against
        _max (int): Maximum value to benchmark against

    Returns:
        pd.Series: Output data series
    """
    return (_s - _min) / (_max - _min)
