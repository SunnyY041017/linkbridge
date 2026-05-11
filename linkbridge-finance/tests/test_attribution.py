"""绩效归因与因子分析模块单元测试"""
import numpy as np
import pandas as pd
import pytest
from linkbridge_finance.attribution import (
    brinson_attribution,
    factor_exposure,
    style_analysis,
    information_coefficient,
)


class TestBrinson:
    def test_brinson_basic(self):
        # 三行业：组合超配行业 1，低配行业 2
        p_w = np.array([0.5, 0.2, 0.3])
        b_w = np.array([0.4, 0.3, 0.3])
        p_ret = np.array([0.10, 0.05, 0.02])
        b_ret = np.array([0.08, 0.06, 0.03])
        b_total = 0.05

        result = brinson_attribution(p_w, b_w, p_ret, b_ret, b_total)

        assert "allocation_effect" in result
        assert "selection_effect" in result
        assert "interaction_effect" in result
        # 总超额应为三效应之和
        total_from_effects = (
            sum(result["allocation_effect"].values())
            + sum(result["selection_effect"].values())
            + sum(result["interaction_effect"].values())
        )
        assert abs(total_from_effects - result["total_excess_pct"]) < 0.01

    def test_brinson_zero_excess(self):
        """当组合与基准完全相同时，所有效应为 0"""
        w = np.array([0.5, 0.5])
        ret = np.array([0.1, 0.1])
        total = 0.1
        result = brinson_attribution(w, w, ret, ret, total)
        assert abs(result["total_excess_pct"]) < 0.01


class TestFactorExposure:
    @pytest.fixture
    def sample_data(self):
        np.random.seed(42)
        n = 100
        dates = pd.date_range("2024-01-01", periods=n, freq="B")
        # 构造一个受两因子影响的股票收益率
        mkt = np.random.normal(0.0005, 0.01, n)
        value = np.random.normal(0.0002, 0.008, n)
        stock = 0.001 + 1.1 * mkt + 0.3 * value + np.random.normal(0, 0.005, n)
        return pd.Series(stock, index=dates), pd.DataFrame({"market": mkt, "value": value}, index=dates)

    def test_factor_exposure_structure(self, sample_data):
        sr, fr = sample_data
        result = factor_exposure(sr, fr)
        assert "alpha" in result
        assert "betas" in result
        assert "r_squared" in result
        assert "residual_volatility" in result
        # 真实 Beta 应接近 1.1 (market) 和 0.3 (value)
        assert 0.5 < result["betas"]["market"] < 1.8
        assert 0.0 < result["betas"]["value"] < 0.8

    def test_factor_exposure_insufficient_data(self):
        sr = pd.Series([0.01, 0.02], index=pd.date_range("2024-01-01", periods=2))
        fr = pd.DataFrame({"mkt": [0.005, 0.006]}, index=sr.index)
        result = factor_exposure(sr, fr)
        assert "error" in result


class TestStyleAnalysis:
    def test_style_analysis_basic(self):
        np.random.seed(42)
        n = 200
        dates = pd.date_range("2024-01-01", periods=n, freq="B")
        # 假设真实风格权重大盘 60% + 小盘 40%
        large = np.random.normal(0.0004, 0.012, n)
        small = np.random.normal(0.0007, 0.018, n)
        stock = 0.6 * large + 0.4 * small + np.random.normal(0, 0.003, n)
        sr = pd.Series(stock, index=dates)
        fr = pd.DataFrame({"large_cap": large, "small_cap": small}, index=dates)

        result = style_analysis(sr, fr)
        assert "weights" in result
        assert "r_squared" in result
        # 权重应接近真实值
        assert 0.3 < result["weights"]["large_cap"] < 0.9
        assert 0.1 < result["weights"]["small_cap"] < 0.7

    def test_style_analysis_empty(self):
        sr = pd.Series(dtype=float)
        fr = pd.DataFrame()
        result = style_analysis(sr, fr)
        assert "error" in result


class TestIC:
    def test_information_coefficient(self):
        np.random.seed(42)
        n = 100
        signals = np.random.normal(0, 1, n)
        forward = signals * 0.5 + np.random.normal(0, 0.5, n)
        result = information_coefficient(signals, forward)
        assert "ic" in result
        assert "rank_ic" in result
        assert abs(result["ic"]) > 0.1  # 应有正向关系

    def test_ic_insufficient_data(self):
        result = information_coefficient(np.array([1.0]), np.array([0.5]))
        assert "error" in result
