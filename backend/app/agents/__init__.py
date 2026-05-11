"""
LinkBridge Agent 集合

六大专业 Agent：
- MarketAgent:  行情分析（Beta/Sharpe/波动率/VaR）
- FundaAgent:   基本面分析（PE/PB/ROE/DCF估值）
- TechnAgent:   技术面分析（MACD/RSI/KDJ/布林带/均线）
- QuantAgent:   量化分析（组合优化/因子暴露/多方法VaR）
- RiskAgent:    风险管理（利率/市场/尾部风险/压力测试）
- SentiAgent:   舆情情绪分析（新闻情绪/市场心理）
"""

from app.agents.market_agent import MarketAgent
from app.agents.funda_agent import FundaAgent
from app.agents.techn_agent import TechnAgent
from app.agents.quant_agent import QuantAgent
from app.agents.risk_agent import RiskAgent
from app.agents.senti_agent import SentiAgent

__all__ = [
    "MarketAgent",
    "FundaAgent",
    "TechnAgent",
    "QuantAgent",
    "RiskAgent",
    "SentiAgent",
]
