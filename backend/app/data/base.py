from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import date
from typing import Optional

import pandas as pd


@dataclass
class StockQuote:
    """单日行情数据"""
    symbol: str
    name: str
    date: date
    open: float
    high: float
    low: float
    close: float
    volume: float
    amount: float
    change_pct: float


@dataclass
class FinancialData:
    """财务指标"""
    symbol: str
    report_date: date
    pe: Optional[float] = None
    pb: Optional[float] = None
    ps: Optional[float] = None
    roe: Optional[float] = None
    roa: Optional[float] = None
    revenue_yoy: Optional[float] = None
    profit_yoy: Optional[float] = None
    total_market_cap: Optional[float] = None
    diluted_eps: Optional[float] = None


class DataProvider(ABC):
    """数据源抽象基类"""

    @abstractmethod
    async def get_stock_history(
        self, symbol: str, start_date: str, end_date: str, period: str = "daily"
    ) -> pd.DataFrame:
        """获取股票历史日线数据"""
        ...

    @abstractmethod
    async def get_stock_info(self, symbol: str) -> dict:
        """获取股票基本信息"""
        ...

    @abstractmethod
    async def get_financial_data(self, symbol: str) -> FinancialData:
        """获取最新财务指标"""
        ...

    @abstractmethod
    async def get_bond_yields(self) -> pd.DataFrame:
        """获取国债收益率曲线"""
        ...

    @abstractmethod
    async def search_stock(self, keyword: str) -> list[dict]:
        """搜索股票"""
        ...
