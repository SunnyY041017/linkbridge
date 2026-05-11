import asyncio
from datetime import date
from functools import partial
from typing import Optional

import akshare as ak
import pandas as pd

from app.data.base import DataProvider, FinancialData


def _run_sync(func, *args, **kwargs):
    """在线程池中运行同步的 AKShare 函数"""
    return asyncio.get_event_loop().run_in_executor(None, partial(func, *args, **kwargs))


class AKShareProvider(DataProvider):
    """AKShare 数据源实现"""

    async def get_stock_history(
        self, symbol: str, start_date: str, end_date: str, period: str = "daily"
    ) -> pd.DataFrame:
        try:
            df = await _run_sync(
                ak.stock_zh_a_hist,
                symbol=symbol,
                period=period,
                start_date=start_date,
                end_date=end_date,
                adjust="qfq",
            )
            if df is None or df.empty:
                return pd.DataFrame()
            df.columns = [c.lower() for c in df.columns]
            return df
        except Exception as e:
            raise RuntimeError(f"获取 {symbol} 历史行情失败: {e}")

    async def get_stock_info(self, symbol: str) -> dict:
        try:
            df = await _run_sync(ak.stock_individual_info_em, symbol=symbol)
            if df is None or df.empty:
                return {}
            info = {}
            for _, row in df.iterrows():
                info[row["item"]] = row["value"]
            return info
        except Exception as e:
            raise RuntimeError(f"获取 {symbol} 基本信息失败: {e}")

    async def get_financial_data(self, symbol: str) -> FinancialData:
        try:
            df = await _run_sync(
                ak.stock_financial_abstract_ths,
                symbol=symbol,
                indicator="按报告期",
            )
            if df is None or df.empty:
                return FinancialData(symbol=symbol, report_date=date.today())

            latest = df.iloc[0] if "报告日期" in df.columns else df.iloc[-1]

            def safe_float(col: str) -> Optional[float]:
                if col in latest.index:
                    v = latest[col]
                    try:
                        return float(v) if pd.notna(v) else None
                    except (ValueError, TypeError):
                        return None
                return None

            return FinancialData(
                symbol=symbol,
                report_date=date.today(),
                pe=safe_float("市盈率"),
                pb=safe_float("市净率"),
                roe=safe_float("净资产收益率"),
                roa=safe_float("总资产收益率"),
                revenue_yoy=safe_float("营业收入同比增长率"),
                profit_yoy=safe_float("归属母公司净利润同比增长率"),
                total_market_cap=safe_float("总市值"),
                diluted_eps=safe_float("基本每股收益"),
            )
        except Exception as e:
            raise RuntimeError(f"获取 {symbol} 财务数据失败: {e}")

    async def get_bond_yields(self, start_date: str = "20240101") -> pd.DataFrame:
        try:
            end_date = date.today().strftime("%Y%m%d")
            df = await _run_sync(
                ak.bond_china_yield,
                start_date=start_date,
                end_date=end_date,
            )
            return df if df is not None else pd.DataFrame()
        except Exception as e:
            raise RuntimeError(f"获取国债收益率数据失败: {e}")

    async def search_stock(self, keyword: str) -> list[dict]:
        try:
            df = await _run_sync(ak.stock_info_a_code_name)
            if df is None or df.empty:
                return []
            mask = df["name"].str.contains(keyword, case=False, na=False)
            results = df[mask].head(10)
            return [{"symbol": r["code"], "name": r["name"]} for _, r in results.iterrows()]
        except Exception as e:
            raise RuntimeError(f"搜索股票失败: {e}")
