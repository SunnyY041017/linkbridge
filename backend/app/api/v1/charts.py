"""图表数据 API — 为前端 ECharts 提供结构化数据"""
import datetime
import uuid

import numpy as np
import pandas as pd
from fastapi import APIRouter, Depends, HTTPException

from app.agents.llm_setup import get_llm_hub
from app.agents.market_agent import MarketAgent
from app.agents.funda_agent import FundaAgent
from app.agents.techn_agent import TechnAgent
from app.agents.risk_agent import RiskAgent
from app.api.deps import get_data_provider

router = APIRouter(tags=["charts"])


def _get_market_agent():
    hub = get_llm_hub()
    return MarketAgent(llm_client=hub.default_client, data_provider=get_data_provider())


def _get_funda_agent():
    hub = get_llm_hub()
    return FundaAgent(llm_client=hub.default_client, data_provider=get_data_provider())


def _get_techn_agent():
    hub = get_llm_hub()
    return TechnAgent(llm_client=hub.default_client, data_provider=get_data_provider())


def _get_risk_agent():
    hub = get_llm_hub()
    return RiskAgent(llm_client=hub.default_client, data_provider=get_data_provider())


@router.get("/chart-data/{symbol}")
async def get_chart_data(symbol: str, days: int = 180):
    """
    获取用于前端 ECharts 渲染的完整图表数据。
    包含：K线、技术指标、风险指标、估值指标、债券风险
    """
    import datetime as dt
    end = dt.date.today().strftime("%Y%m%d")
    start = (dt.date.today() - dt.timedelta(days=days + 60)).strftime("%Y%m%d")

    provider = get_data_provider()
    df = await provider.get_stock_history(symbol, start, end)
    if df.empty:
        raise HTTPException(status_code=404, detail=f"未找到 {symbol} 的数据")

    data_source = df["_source"].iloc[0] if "_source" in df.columns else "live"

    close = pd.to_numeric(df["收盘"], errors="coerce")
    high = pd.to_numeric(df["最高"], errors="coerce")
    low = pd.to_numeric(df["最低"], errors="coerce")
    open_p = pd.to_numeric(df["开盘"], errors="coerce")
    volume = pd.to_numeric(df["成交量"], errors="coerce")

    dates = df["日期"].astype(str).tolist() if "日期" in df.columns else [str(d)[:10] for d in df.index]

    # K 线数据
    kline = []
    for i in range(len(df)):
        kline.append([
            float(open_p.iloc[i]) if not pd.isna(open_p.iloc[i]) else None,
            float(close.iloc[i]) if not pd.isna(close.iloc[i]) else None,
            float(low.iloc[i]) if not pd.isna(low.iloc[i]) else None,
            float(high.iloc[i]) if not pd.isna(high.iloc[i]) else None,
        ])

    vol_data = [float(volume.iloc[i]) if not pd.isna(volume.iloc[i]) else 0 for i in range(len(df))]

    # 均线
    from linkbridge_finance.technical import sma
    ma5 = sma(close, 5).round(2).fillna(0).tolist() if len(close) >= 5 else []
    ma10 = sma(close, 10).round(2).fillna(0).tolist() if len(close) >= 10 else []
    ma20 = sma(close, 20).round(2).fillna(0).tolist() if len(close) >= 20 else []

    # MACD
    macd_data = {"dif": [], "dea": [], "hist": []}
    if len(close) >= 26:
        from linkbridge_finance.technical import macd
        macd_df = macd(close)
        macd_data = {
            "dif": macd_df["dif"].round(3).fillna(0).tolist(),
            "dea": macd_df["dea"].round(3).fillna(0).tolist(),
            "hist": macd_df["macd_hist"].round(3).fillna(0).tolist(),
        }

    # RSI
    rsi_vals = []
    if len(close) >= 14:
        from linkbridge_finance.technical import rsi
        rsi_vals = rsi(close).round(1).fillna(50).tolist()

    # 布林带
    bb = {"upper": [], "mid": [], "lower": []}
    if len(close) >= 20:
        from linkbridge_finance.technical import bollinger_bands
        bb_df = bollinger_bands(close)
        bb = {
            "upper": bb_df["upper"].round(2).fillna(0).tolist(),
            "mid": bb_df["middle"].round(2).fillna(0).tolist(),
            "lower": bb_df["lower"].round(2).fillna(0).tolist(),
        }

    # 风险收益指标
    returns = close.pct_change().dropna()
    from linkbridge_finance.risk import (
        annualized_return, annualized_volatility, sharpe_ratio,
        beta, max_drawdown, value_at_risk_historical,
    )

    risk = {
        "annualized_return": round(annualized_return(returns) * 100, 2) if len(returns) > 20 else None,
        "annualized_volatility": round(annualized_volatility(returns) * 100, 2) if len(returns) > 5 else None,
        "sharpe_ratio": round(sharpe_ratio(returns), 2) if len(returns) > 20 else None,
        "max_drawdown": round(max_drawdown(returns) * 100, 2) if len(returns) > 5 else None,
        "var_95": round(value_at_risk_historical(returns, 0.95) * 100, 2) if len(returns) > 20 else None,
        "beta": None,
    }

    # Beta（针对沪深300）
    try:
        bench_df = await provider.get_stock_history("000300", start, end)
        bench_close = pd.to_numeric(bench_df["收盘"], errors="coerce")
        bench_ret = bench_close.pct_change().dropna()
        if len(returns) > 20 and len(bench_ret) > 20:
            risk["beta"] = round(beta(returns, bench_ret), 2)
    except Exception:
        pass

    # 估值面
    fin_data = await provider.get_financial_data(symbol)
    info = await provider.get_stock_info(symbol)
    price = float(info.get("最新价", float(close.iloc[-1])))
    eps = fin_data.diluted_eps if fin_data else 0

    from linkbridge_finance.valuation import pe_ratio, pb_ratio, peg_ratio
    valuation = {
        "pe": round(pe_ratio(price, eps), 2) if eps and eps > 0 else None,
        "pb": round(pb_ratio(price, float(info.get("每股净资产", 0))), 2) if info.get("每股净资产") else None,
        "roe": round(fin_data.roe, 2) if fin_data and fin_data.roe else None,
        "roa": round(fin_data.roa, 2) if fin_data and fin_data.roa else None,
        "revenue_yoy": round(fin_data.revenue_yoy, 2) if fin_data and fin_data.revenue_yoy else None,
        "profit_yoy": round(fin_data.profit_yoy, 2) if fin_data and fin_data.profit_yoy else None,
    }

    # 债券风险指标
    from linkbridge_finance.rates import macaulay_duration, modified_duration, convexity, bond_price
    face_value = 100.0
    coupon_rate = 0.03
    periods = 10
    ytm = 0.035
    cash_flows = [coupon_rate * face_value] * (periods - 1) + [coupon_rate * face_value + face_value]
    times = list(range(1, periods + 1))
    bond_risk = {
        "macaulay_duration": round(macaulay_duration(cash_flows, times, ytm), 2),
        "modified_duration": round(modified_duration(cash_flows, times, ytm), 2),
        "convexity": round(convexity(cash_flows, times, ytm), 2),
        "current_price": round(bond_price(face_value, coupon_rate, periods, ytm), 2),
    }

    # 利率敏感性曲线（YTM ±200bp 的价格）
    ytm_range = np.linspace(0.015, 0.055, 40)
    price_curve = [
        {"ytm": round(y * 100, 2), "price": round(bond_price(face_value, coupon_rate, periods, y), 2)}
        for y in ytm_range
    ]

    return {
        "symbol": symbol,
        "data_source": data_source,
        "dates": dates[-days:],
        "kline": kline[-days:],
        "volume": vol_data[-days:],
        "ma5": ma5[-days:] if ma5 else [],
        "ma10": ma10[-days:] if ma10 else [],
        "ma20": ma20[-days:] if ma20 else [],
        "macd": {k: v[-days:] if v else [] for k, v in macd_data.items()},
        "rsi": rsi_vals[-days:] if rsi_vals else [],
        "bollinger": {k: v[-days:] if v else [] for k, v in bb.items()},
        "risk": risk,
        "valuation": valuation,
        "bond_risk": bond_risk,
        "price_curve": price_curve,
    }
