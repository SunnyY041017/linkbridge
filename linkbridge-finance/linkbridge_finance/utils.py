import numpy as np
import pandas as pd


def safe_float_series(series: pd.Series) -> pd.Series:
    """安全转换为 float，非数值置为 NaN"""
    return pd.to_numeric(series, errors="coerce")


def annualize_return(total_return: float, days: int) -> float:
    """年化收益率"""
    if days <= 0 or total_return <= -1:
        return np.nan
    return (1 + total_return) ** (365 / days) - 1


def annualize_vol(daily_vol: float) -> float:
    """年化波动率"""
    return daily_vol * np.sqrt(252)


def rolling_window(arr: np.ndarray, window: int):
    """生成滑动窗口"""
    shape = arr.shape[:-1] + (arr.shape[-1] - window + 1, window)
    strides = arr.strides + (arr.strides[-1],)
    return np.lib.stride_tricks.sliding_window_view(arr, window)


def risk_free_rate() -> float:
    """默认无风险利率（中国十年期国债约 2.5%）"""
    return 0.025
