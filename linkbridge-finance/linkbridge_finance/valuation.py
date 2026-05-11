"""
估值指标模块

PE、PB、PS、PEG、股息率、自由现金流收益率、DCF 估值模型
"""

import numpy as np
import pandas as pd


def pe_ratio(price: float, earnings_per_share: float) -> float:
    """
    市盈率 PE = 股价 / 每股收益

    注意：EPS <= 0 时 PE 无意义，返回 NaN
    """
    if earnings_per_share <= 0:
        return np.nan
    return float(price / earnings_per_share)


def pb_ratio(price: float, book_value_per_share: float) -> float:
    """
    市净率 PB = 股价 / 每股净资产
    """
    if book_value_per_share <= 0:
        return np.nan
    return float(price / book_value_per_share)


def ps_ratio(price: float, revenue_per_share: float) -> float:
    """
    市销率 PS = 股价 / 每股营收
    """
    if revenue_per_share <= 0:
        return np.nan
    return float(price / revenue_per_share)


def peg_ratio(pe: float, earnings_growth_rate: float) -> float:
    """
    PEG = PE / 盈利增长率(%)

    通常 PEG < 1 被认为低估
    """
    if earnings_growth_rate <= 0:
        return np.nan
    return float(pe / (earnings_growth_rate * 100))


def dividend_yield(
    dividend_per_share: float,
    price: float,
) -> float:
    """股息率 = 年度每股分红 / 股价"""
    if price <= 0:
        return 0.0
    return float(dividend_per_share / price)


def free_cash_flow_yield(
    free_cash_flow: float,
    market_cap: float,
) -> float:
    """自由现金流收益率 = FCF / 总市值"""
    if market_cap <= 0:
        return np.nan
    return float(free_cash_flow / market_cap)


def roe(earnings: float, equity: float) -> float:
    """净资产收益率 ROE = 净利润 / 净资产"""
    if equity == 0:
        return np.nan
    return float(earnings / equity)


def roa(earnings: float, total_assets: float) -> float:
    """总资产收益率 ROA = 净利润 / 总资产"""
    if total_assets == 0:
        return np.nan
    return float(earnings / total_assets)


def ev_to_ebitda(
    enterprise_value: float,
    ebitda: float,
) -> float:
    """
    EV/EBITDA — 企业价值倍数。

    适用于跨行业估值对比，排除了资本结构和折旧政策影响。
    """
    if ebitda <= 0:
        return np.nan
    return float(enterprise_value / ebitda)


def dcf_valuation(
    free_cash_flows: list[float],
    terminal_growth_rate: float = 0.03,
    discount_rate: float = 0.10,
    shares_outstanding: float = 1.0,
    net_debt: float = 0.0,
) -> dict:
    """
    两阶段 DCF 估值模型。

    Stage 1: 显式预测期（用给定 FCF）
    Stage 2: 终值期（永续增长模型）

    Parameters:
        free_cash_flows:      预测期各年自由现金流
        terminal_growth_rate: 永续增长率
        discount_rate:        折现率 (WACC)
        shares_outstanding:   总股本（亿股）
        net_debt:             净债务

    Returns:
        {
            'enterprise_value': 企业价值,
            'equity_value':     股权价值,
            'fair_price':       每股公允价值,
            'pv_cashflows':     预测期现金流现值,
            'terminal_value':   终值,
            'pv_terminal':      终值现值
        }
    """
    # Stage 1: 折现预测期现金流
    pv_cashflows = 0.0
    for t, fcf in enumerate(free_cash_flows, 1):
        pv_cashflows += fcf / (1 + discount_rate) ** t

    last_fcf = free_cash_flows[-1] if free_cash_flows else 0

    # Stage 2: 终值 = FCF_n × (1+g) / (r-g)
    terminal_value = last_fcf * (1 + terminal_growth_rate) / (
        discount_rate - terminal_growth_rate
    )
    pv_terminal = terminal_value / (1 + discount_rate) ** len(free_cash_flows)

    enterprise_value = pv_cashflows + pv_terminal
    equity_value = enterprise_value - net_debt
    fair_price = equity_value / shares_outstanding if shares_outstanding > 0 else np.nan

    return {
        "enterprise_value": round(enterprise_value, 2),
        "equity_value": round(equity_value, 2),
        "fair_price": round(fair_price, 2),
        "pv_cashflows": round(pv_cashflows, 2),
        "terminal_value": round(terminal_value, 2),
        "pv_terminal": round(pv_terminal, 2),
    }


def gordon_growth_model(
    dividend: float,
    growth_rate: float,
    required_return: float,
) -> float:
    """
    戈登增长模型 — 适用于稳定派息的成熟公司。

    P = D / (r - g)

    Parameters:
        dividend:        预期下一年每股分红
        growth_rate:     永续股息增长率
        required_return: 要求回报率

    Returns:
        理论股价
    """
    if required_return <= growth_rate:
        return np.nan
    return float(dividend / (required_return - growth_rate))


def net_net_working_capital(
    current_assets: float,
    total_liabilities: float,
    shares_outstanding: float,
) -> float:
    """
    Graham 净净营运资本法 — 深度价值估值。

    NCAV = (流动资产 - 总负债) / 总股本

    当股价 < NCAV × 2/3 时被认为是极低估。
    """
    ncav = current_assets - total_liabilities
    if shares_outstanding <= 0:
        return np.nan
    return float(ncav / shares_outstanding)


def earning_yield(pe: float) -> float:
    """盈利收益率 = 1 / PE"""
    if pe <= 0:
        return np.nan
    return float(1 / pe)
