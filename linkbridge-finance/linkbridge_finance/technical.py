"""
技术指标模块

MACD、RSI、KDJ、布林带、均线系统 (MA/EMA)、ATR
"""

import numpy as np
import pandas as pd


def sma(series: pd.Series, period: int) -> pd.Series:
    """简单移动平均 (SMA)"""
    return series.rolling(window=period).mean()


def ema(series: pd.Series, period: int) -> pd.Series:
    """指数移动平均 (EMA)"""
    return series.ewm(span=period, adjust=False).mean()


def macd(
    close: pd.Series,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
) -> pd.DataFrame:
    """
    MACD 指标。

    Returns:
        DataFrame with columns: ['dif', 'dea', 'macd_hist']
    """
    ema_fast = ema(close, fast)
    ema_slow = ema(close, slow)
    dif = ema_fast - ema_slow
    dea = ema(dif, signal)
    macd_hist = (dif - dea) * 2  # 红绿柱 ×2 是国内惯例
    return pd.DataFrame({"dif": dif, "dea": dea, "macd_hist": macd_hist})


def rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """
    RSI 相对强弱指标 (Wilder's smoothing)。

    RSI = 100 - 100 / (1 + RS)
    RS = 平均涨幅 / 平均跌幅
    """
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)

    avg_gain = gain.ewm(alpha=1 / period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - 100 / (1 + rs)


def kdj(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    n: int = 9,
    m1: int = 3,
    m2: int = 3,
) -> pd.DataFrame:
    """
    KDJ 随机指标。

    Returns:
        DataFrame with columns: ['k', 'd', 'j']
    """
    lowest_low = low.rolling(window=n).min()
    highest_high = high.rolling(window=n).max()

    rsv = (close - lowest_low) / (highest_high - lowest_low) * 100
    rsv = rsv.fillna(50)

    k = rsv.ewm(alpha=1 / m1, adjust=False).mean()
    d = k.ewm(alpha=1 / m2, adjust=False).mean()
    j = 3 * k - 2 * d

    return pd.DataFrame({"k": k, "d": d, "j": j})


def bollinger_bands(
    close: pd.Series,
    period: int = 20,
    std_dev: float = 2.0,
) -> pd.DataFrame:
    """
    布林带。

    Returns:
        DataFrame with columns: ['middle', 'upper', 'lower', 'width', 'pct_b']
    """
    middle = sma(close, period)
    std = close.rolling(window=period).std()
    upper = middle + std_dev * std
    lower = middle - std_dev * std
    width = (upper - lower) / middle * 100
    pct_b = (close - lower) / (upper - lower)
    return pd.DataFrame(
        {"middle": middle, "upper": upper, "lower": lower, "width": width, "pct_b": pct_b}
    )


def atr(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    period: int = 14,
) -> pd.Series:
    """
    ATR 平均真实波幅 (Wilder's smoothing)。

    TR = max(high - low, |high - prev_close|, |low - prev_close|)
    """
    prev_close = close.shift(1)
    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / period, adjust=False).mean()


def obv(close: pd.Series, volume: pd.Series) -> pd.Series:
    """
    OBV 能量潮 — 通过成交量变化判断价格趋势。

    OBV_t = OBV_{t-1} + volume_t × sign(close_t - close_{t-1})
    """
    direction = np.sign(close.diff())
    obv_series = (volume * direction).cumsum()
    return obv_series


def ma_cross_signal(
    close: pd.Series,
    fast_period: int = 5,
    slow_period: int = 20,
) -> pd.Series:
    """
    均线交叉信号。

    Returns:
        1=金叉 (买入), -1=死叉 (卖出), 0=无信号
    """
    fast_ma = sma(close, fast_period)
    slow_ma = sma(close, slow_period)

    fast_prev = fast_ma.shift(1)
    slow_prev = slow_ma.shift(1)

    golden_cross = (fast_prev <= slow_prev) & (fast_ma > slow_ma)
    death_cross = (fast_prev >= slow_prev) & (fast_ma < slow_ma)

    signal = pd.Series(0, index=close.index, dtype=int)
    signal[golden_cross] = 1
    signal[death_cross] = -1
    return signal


def support_resistance(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    window: int = 20,
) -> pd.DataFrame:
    """
    支撑/阻力位 — 基于近期高低点。

    Returns:
        DataFrame with columns: ['support', 'resistance', 'pivot']
    """
    resistance = high.rolling(window=window).max()
    support = low.rolling(window=window).min()
    pivot = (resistance + support + close) / 3
    return pd.DataFrame({"support": support, "resistance": resistance, "pivot": pivot})


def williams_r(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    period: int = 14,
) -> pd.Series:
    """
    威廉指标 (Williams %R)。

    %R = (highest_high - close) / (highest_high - lowest_low) × -100
    """
    highest = high.rolling(window=period).max()
    lowest = low.rolling(window=period).min()
    return (highest - close) / (highest - lowest) * -100


def cci(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    period: int = 20,
) -> pd.Series:
    """
    CCI 商品通道指数。

    CCI = (TP - SMA(TP)) / (0.015 × mean_deviation)
    """
    tp = (high + low + close) / 3
    tp_sma = sma(tp, period)
    mean_dev = tp.rolling(window=period).apply(lambda x: np.abs(x - x.mean()).mean())
    return (tp - tp_sma) / (0.015 * mean_dev)
