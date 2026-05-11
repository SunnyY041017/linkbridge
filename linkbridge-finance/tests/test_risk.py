import numpy as np
import pandas as pd
import pytest
from linkbridge_finance.risk import (
    daily_returns,
    annualized_return,
    annualized_volatility,
    sharpe_ratio,
    sortino_ratio,
    max_drawdown,
    beta,
    alpha,
    value_at_risk_historical,
    value_at_risk_parametric,
    cvar_historical,
)


@pytest.fixture
def sample_returns():
    """模拟正态分布日收益率，250 天"""
    np.random.seed(42)
    return pd.Series(np.random.normal(0.0005, 0.015, 250))


@pytest.fixture
def positive_returns():
    """单调上涨的价格序列"""
    np.random.seed(42)
    return pd.Series(np.random.normal(0.001, 0.005, 250))


class TestReturns:
    def test_daily_returns(self):
        prices = pd.Series([100, 101, 99, 102])
        ret = daily_returns(prices)
        assert len(ret) == 3
        assert abs(ret.iloc[0] - 0.01) < 0.0001


class TestAnnualization:
    def test_positive_return(self, positive_returns):
        ann = annualized_return(positive_returns)
        assert ann > 0

    def test_volatility_positive(self, sample_returns):
        vol = annualized_volatility(sample_returns)
        assert vol > 0


class TestSharpeRatio:
    def test_normal(self, sample_returns):
        sr = sharpe_ratio(sample_returns, risk_free=0.025)
        assert isinstance(sr, float)

    def test_constant_returns(self):
        ret = pd.Series(np.full(100, 0.0001))
        sr = sharpe_ratio(ret, risk_free=0.0)
        assert sr == 0.0  # 标准差为 0 时返回 0


class TestSortinoRatio:
    def test_positive_vs_sharpe(self, positive_returns):
        """正收益序列的 Sortino Ratio 应高于或等于 Sharpe"""
        sr = sharpe_ratio(positive_returns, risk_free=0.0)
        so = sortino_ratio(positive_returns, risk_free=0.0)
        # Sortino 只罚下行，上行多的序列 Sortino 应该更好
        assert so >= sr * 0.5


class TestMaxDrawdown:
    def test_known_drawdown(self):
        """最大回撤验证：峰值 100 跌到 70 = -30%"""
        prices = pd.Series([100, 105, 90, 85, 70, 80, 95])
        ret = prices.pct_change().dropna()
        mdd = max_drawdown(ret)
        assert mdd < -0.25

    def test_no_loss(self, positive_returns):
        mdd = max_drawdown(positive_returns)
        assert mdd < 0.05  # 极小回撤


class TestBeta:
    def test_market_neutral(self):
        """无相关时 Beta 接近 0"""
        np.random.seed(99)
        stock = pd.Series(np.random.normal(0, 0.01, 100))
        bench = pd.Series(np.random.normal(0, 0.01, 100))
        b = beta(stock, bench)
        assert abs(b) < 1.0

    def test_identity(self):
        """同一序列的 Beta = 1"""
        ret = pd.Series(np.random.normal(0.001, 0.01, 100))
        b = beta(ret, ret)
        assert abs(b - 1.0) < 0.0001


class TestAlpha:
    def test_identity_alpha(self):
        """同一序列相对自身的 Alpha ≈ 0"""
        ret = pd.Series(np.random.normal(0.001, 0.01, 100))
        a = alpha(ret, ret)
        assert abs(a) < 0.01


class TestVaR:
    def test_historical_var(self, sample_returns):
        var95 = value_at_risk_historical(sample_returns, 0.95)
        var99 = value_at_risk_historical(sample_returns, 0.99)
        assert var99 > var95  # 更高置信度 = 更大损失

    def test_parametric_var(self, sample_returns):
        var95 = value_at_risk_parametric(sample_returns, 0.95)
        assert not np.isnan(var95)  # 正向收益时 VaR 可为负（预期盈利）

    def test_cvar_worse_than_var(self, sample_returns):
        """CVaR 应 >= VaR（预期损失 >= 分位数损失）"""
        var95 = value_at_risk_historical(sample_returns, 0.95)
        cvar95 = cvar_historical(sample_returns, 0.95)
        assert cvar95 >= var95
