import numpy as np
import pytest
from linkbridge_finance.valuation import (
    pe_ratio,
    pb_ratio,
    ps_ratio,
    peg_ratio,
    roe,
    roa,
    dividend_yield,
    free_cash_flow_yield,
    dcf_valuation,
    gordon_growth_model,
    ev_to_ebitda,
)


class TestMultiples:
    def test_pe_normal(self):
        pe = pe_ratio(100, 5)
        assert abs(pe - 20) < 0.01

    def test_pe_negative_eps(self):
        pe = pe_ratio(100, -1)
        assert np.isnan(pe)

    def test_pb(self):
        pb = pb_ratio(100, 20)
        assert abs(pb - 5) < 0.01

    def test_ps(self):
        ps = ps_ratio(50, 10)
        assert abs(ps - 5) < 0.01

    def test_peg(self):
        peg = peg_ratio(20, 0.30)  # 30% 增长率
        assert 0.5 < peg < 1.0  # ≈ 0.67

    def test_ev_ebitda(self):
        ev = ev_to_ebitda(1000, 100)
        assert abs(ev - 10) < 0.01


class TestProfitability:
    def test_roe(self):
        r = roe(100, 500)
        assert abs(r - 0.20) < 0.001

    def test_roa(self):
        r = roa(100, 1000)
        assert abs(r - 0.10) < 0.001


class TestYields:
    def test_dividend_yield(self):
        dy = dividend_yield(3, 100)
        assert abs(dy - 0.03) < 0.001

    def test_fcf_yield(self):
        fcfy = free_cash_flow_yield(50, 500)
        assert abs(fcfy - 0.10) < 0.001


class TestDCF:
    def test_dcf_valuation(self):
        fcf = [10, 12, 14, 16, 18]
        result = dcf_valuation(
            fcf,
            terminal_growth_rate=0.03,
            discount_rate=0.10,
            shares_outstanding=10,
            net_debt=0,
        )
        assert result["fair_price"] > 0
        assert result["enterprise_value"] > result["terminal_value"] * 0.3

    def test_dcf_high_growth(self):
        """高增长公司终值占比更大"""
        fcf = [1, 2, 5, 10, 20]
        result = dcf_valuation(fcf, terminal_growth_rate=0.04, discount_rate=0.10)
        assert result["pv_terminal"] > result["pv_cashflows"]


class TestGordon:
    def test_gordon(self):
        p = gordon_growth_model(3, 0.03, 0.10)
        assert p > 30

    def test_invalid(self):
        """r <= g 时返回 NaN"""
        p = gordon_growth_model(3, 0.10, 0.08)
        assert np.isnan(p)
