import numpy as np
import pytest
from linkbridge_finance.rates import (
    macaulay_duration,
    modified_duration,
    convexity,
    bond_price,
    price_impact_duration,
    bond_portfolio_metrics,
)


class TestDuration:
    def test_zero_coupon_bond_duration(self):
        """零息债券的麦考利久期应等于剩余期限"""
        face = 100
        ytm = 0.05
        t = 5
        cf = np.array([0, 0, 0, 0, face])
        times = np.array([1, 2, 3, 4, 5])
        dur = macaulay_duration(cf, times, ytm)
        assert abs(dur - t) < 0.01

    def test_perpetual_duration(self):
        """永续债券久期近似 (1+y)/y"""
        ytm = 0.05
        n_periods = 200
        coupon = 5
        cf = np.full(n_periods, coupon)
        cf[-1] += 100  # face value at the end makes it non-perpetual, but long enough
        times = np.arange(1, n_periods + 1)
        dur = macaulay_duration(cf, times, ytm)
        expected = (1 + ytm) / ytm  # 理论值 21
        assert dur > 15  # 200 期足够接近

    def test_modified_duration(self):
        """修正久期 = 麦考利久期 / (1 + ytm)"""
        ytm = 0.06
        cf = np.array([6, 6, 6, 6, 106])
        times = np.array([1, 2, 3, 4, 5])
        d_mod = modified_duration(cf, times, ytm)
        d_mac = macaulay_duration(cf, times, ytm)
        assert abs(d_mod - d_mac / (1 + ytm)) < 0.0001


class TestConvexity:
    def test_positive_convexity(self):
        """普通付息债券凸性应为正"""
        ytm = 0.05
        cf = np.array([5, 5, 5, 5, 105])
        times = np.array([1, 2, 3, 4, 5])
        conv = convexity(cf, times, ytm)
        assert conv > 0


class TestBondPrice:
    def test_par_bond(self):
        """票面利率 = YTM 时，价格 = 面值"""
        price = bond_price(100, 0.05, 10, 0.05)
        assert abs(price - 100) < 0.01

    def test_premium_bond(self):
        """票面利率 > YTM 时，价格 > 面值"""
        price = bond_price(100, 0.08, 10, 0.05)
        assert price > 100

    def test_discount_bond(self):
        """票面利率 < YTM 时，价格 < 面值"""
        price = bond_price(100, 0.03, 10, 0.05)
        assert price < 100


class TestPriceImpact:
    def test_rate_hike_impact(self):
        """利率上升 1%，债券价格应下跌"""
        ytm = 0.05
        cf = np.array([5, 5, 5, 5, 105])
        times = np.array([1, 2, 3, 4, 5])
        d_mod = modified_duration(cf, times, ytm)
        conv = convexity(cf, times, ytm)
        impact = price_impact_duration(d_mod, conv, 0.01)
        assert impact < 0  # 价格下跌

    def test_rate_cut_impact(self):
        """利率下降，债券价格应上涨"""
        ytm = 0.05
        cf = np.array([5, 5, 5, 5, 105])
        times = np.array([1, 2, 3, 4, 5])
        d_mod = modified_duration(cf, times, ytm)
        conv = convexity(cf, times, ytm)
        impact = price_impact_duration(d_mod, conv, -0.01)
        assert impact > 0  # 价格上涨


class TestPortfolioMetrics:
    def test_portfolio(self):
        bonds = [
            {
                "market_value": 500000,
                "cashflows": [5, 5, 5, 5, 105],
                "times": [1, 2, 3, 4, 5],
            },
            {
                "market_value": 300000,
                "cashflows": [3, 3, 103],
                "times": [1, 2, 3],
            },
        ]
        yields = [0.05, 0.04]
        result = bond_portfolio_metrics(bonds, yields)
        assert "portfolio_duration" in result
        assert "portfolio_convexity" in result
        assert "weighted_ytm" in result
        assert result["portfolio_duration"] > 0
        assert result["portfolio_convexity"] > 0
