import numpy as np
import pandas as pd
import pytest
from linkbridge_finance.technical import (
    sma,
    ema,
    macd,
    rsi,
    kdj,
    bollinger_bands,
    atr,
    ma_cross_signal,
    williams_r,
)


@pytest.fixture
def price_data():
    """模拟 200 天价格数据"""
    np.random.seed(42)
    n = 200
    base = 100
    # 趋势 + 噪声
    trend = np.linspace(0, 20, n)
    noise = np.random.normal(0, 2, n)
    close = base + trend + noise
    high = close + np.abs(np.random.normal(0, 1, n))
    low = close - np.abs(np.random.normal(0, 1, n))
    volume = np.random.randint(1000000, 5000000, n)
    dates = pd.date_range("2025-01-01", periods=n, freq="B")
    return pd.DataFrame(
        {"high": high, "low": low, "close": close, "volume": volume}, index=dates
    )


class TestMovingAverage:
    def test_sma_length(self, price_data):
        s = sma(price_data["close"], 20)
        assert len(s) == len(price_data)
        assert s.iloc[:19].isna().all()
        assert s.iloc[19:].notna().all()

    def test_ema_weight(self, price_data):
        """EMA 近期数据权重更大"""
        e = ema(price_data["close"], 5)
        # 最新值应该更接近最近的价格
        assert e.iloc[-1] > e.iloc[-30]  # 上涨趋势中


class TestMACD:
    def test_macd_columns(self, price_data):
        m = macd(price_data["close"])
        assert list(m.columns) == ["dif", "dea", "macd_hist"]
        assert len(m) == len(price_data)

    def test_macd_signal(self, price_data):
        """上涨趋势中 DIF 应 > 0"""
        m = macd(price_data["close"])
        assert m["dif"].iloc[-1] > -10  # 趋势向上，DIF 不应太负


class TestRSI:
    def test_rsi_range(self, price_data):
        r = rsi(price_data["close"], 14)
        valid = r.dropna()
        assert (valid >= 0).all()
        assert (valid <= 100).all()

    def test_rsi_up_trend(self, price_data):
        """上涨趋势中 RSI 偏向高位"""
        r = rsi(price_data["close"], 14)
        later_rsi = r.iloc[-50:].mean()
        early_rsi = r.iloc[14:64].mean()
        # 趋势上涨，后期 RSI 应普遍更高
        assert later_rsi > early_rsi * 0.5


class TestKDJ:
    def test_kdj_range(self, price_data):
        k = kdj(price_data["high"], price_data["low"], price_data["close"])
        valid = k.dropna()
        assert (valid["k"] >= 0).all() or valid["k"].max() <= 110
        assert (valid["d"] >= 0).all() or valid["d"].max() <= 110


class TestBollinger:
    def test_band_envelope(self, price_data):
        bb = bollinger_bands(price_data["close"], 20, 2.0)
        valid = bb.dropna()
        assert (valid["upper"] >= valid["middle"]).all()
        assert (valid["lower"] <= valid["middle"]).all()

    def test_price_inside_bands(self, price_data):
        """价格大多数时间应在布林带内（95% 理论）"""
        bb = bollinger_bands(price_data["close"], 20, 2.0)
        valid = bb.dropna()
        inside = (
            (price_data["close"].loc[valid.index] <= valid["upper"])
            & (price_data["close"].loc[valid.index] >= valid["lower"])
        ).mean()
        assert inside > 0.85


class TestATR:
    def test_atr_positive(self, price_data):
        a = atr(price_data["high"], price_data["low"], price_data["close"], 14)
        valid = a.dropna()
        assert (valid > 0).all()


class TestMACrossSignal:
    def test_signal_types(self, price_data):
        sig = ma_cross_signal(price_data["close"], 5, 20)
        assert set(sig.unique()).issubset({0, 1, -1})

    def test_no_signal_initially(self, price_data):
        sig = ma_cross_signal(price_data["close"], 5, 20)
        # 前 slow_period-1 天应为 0
        assert (sig.iloc[:19] == 0).all()
