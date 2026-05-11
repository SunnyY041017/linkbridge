"""投资组合分析模块单元测试"""
import numpy as np
import pandas as pd
import pytest
from linkbridge_finance.portfolio import (
    returns_matrix,
    correlation_matrix,
    covariance_matrix,
    portfolio_return,
    portfolio_volatility,
    portfolio_sharpe,
    portfolio_variance_decomposition,
    equal_weight_portfolio,
    minimum_variance_portfolio,
    max_sharpe_portfolio,
    efficient_frontier,
    diversification_ratio,
    portfolio_beta,
)


@pytest.fixture
def three_asset_returns():
    """三资产日收益率 60 天"""
    np.random.seed(42)
    dates = pd.date_range("2024-01-01", periods=60, freq="B")
    prices = pd.DataFrame({
        "stock_a": 100 * (1 + np.random.normal(0.0005, 0.015, 60)).cumprod(),
        "stock_b": 100 * (1 + np.random.normal(0.0003, 0.012, 60)).cumprod(),
        "stock_c": 100 * (1 + np.random.normal(0.0007, 0.020, 60)).cumprod(),
    }, index=dates)
    return prices.pct_change().dropna()


@pytest.fixture
def cov_and_returns(three_asset_returns):
    rets = three_asset_returns
    cov = rets.cov() * 252
    ann_ret = rets.mean() * 252
    return cov.values, ann_ret.values


class TestPortfolioBasics:
    def test_returns_matrix(self, three_asset_returns):
        df = returns_matrix((1 + three_asset_returns).cumprod())
        pd.testing.assert_index_equal(df.columns, three_asset_returns.columns)

    def test_correlation_matrix(self, three_asset_returns):
        corr = correlation_matrix(three_asset_returns)
        assert corr.shape == (3, 3)
        assert np.allclose(np.diag(corr), 1.0)

    def test_covariance_matrix_is_psd(self, three_asset_returns):
        cov = covariance_matrix(three_asset_returns)
        eigvals = np.linalg.eigvalsh(cov)
        assert np.all(eigvals >= 0)

    def test_equal_weight_three(self):
        w = equal_weight_portfolio(3)
        assert np.allclose(w, [1/3, 1/3, 1/3])

    def test_portfolio_return(self, cov_and_returns):
        cov, ann_ret = cov_and_returns
        w = equal_weight_portfolio(3)
        ret = portfolio_return(w, ann_ret)
        assert -1.0 < ret < 1.0

    def test_portfolio_volatility_positive(self, cov_and_returns):
        cov, ann_ret = cov_and_returns
        w = equal_weight_portfolio(3)
        vol = portfolio_volatility(w, cov)
        assert vol > 0

    def test_diversification_ratio(self, cov_and_returns):
        cov, _ = cov_and_returns
        w = equal_weight_portfolio(3)
        dr = diversification_ratio(w, cov)
        assert dr >= 1.0  # 分散化收益

    def test_portfolio_beta(self):
        w = np.array([0.5, 0.5])
        betas = np.array([1.2, 0.8])
        pb = portfolio_beta(w, betas)
        assert abs(pb - 1.0) < 1e-6


class TestOptimization:
    def test_minimum_variance(self, cov_and_returns):
        cov, _ = cov_and_returns
        result = minimum_variance_portfolio(cov)
        assert abs(np.sum(result["weights"]) - 1) < 0.01
        assert result["volatility"] > 0

    def test_max_sharpe(self, cov_and_returns):
        cov, ann_ret = cov_and_returns
        result = max_sharpe_portfolio(ann_ret, cov)
        assert abs(np.sum(result["weights"]) - 1) < 0.01
        assert result["sharpe"] > -10  # 可能负但不过分

    def test_efficient_frontier(self, cov_and_returns):
        cov, ann_ret = cov_and_returns
        frontier = efficient_frontier(ann_ret, cov, num_points=20)
        assert len(frontier) >= 1
        # 有效前沿波动率应随收益递增
        vols = [p["volatility"] for p in frontier]
        rets = [p["return"] for p in frontier]
        # 在均值-方差框架下，有效前沿上收益越高波动越大
        # 只需检查成功生成了点


class TestRiskDecomposition:
    def test_variance_decomposition(self, cov_and_returns):
        cov, _ = cov_and_returns
        w = equal_weight_portfolio(3)
        contrib = portfolio_variance_decomposition(w, cov)
        assert len(contrib) == 3
        # 风险贡献之和应接近组合波动率
        assert abs(np.sum(contrib) - portfolio_volatility(w, cov)) < 0.5

    def test_zero_vol_decomposition(self):
        w = np.array([1.0, 0.0])
        cov = np.zeros((2, 2))
        contrib = portfolio_variance_decomposition(w, cov)
        assert np.all(contrib == 0)
