"""
灵桥金融指标计算引擎 (LinkBridge Financial Indicator Engine)

六大模块：
- rates:       利率风险指标（久期、修正久期、凸性、DV01）
- risk:        风险收益指标（Sharpe、Sortino、Beta、Alpha、最大回撤、VaR/CVaR）
- valuation:   估值指标（PE、PB、PS、PEG、DCF、Gordon Growth）
- technical:   技术指标（MACD、KDJ、RSI、布林带、均线系统）
- portfolio:   组合分析（有效前沿、最小方差、最大夏普、风险平价）
- attribution: 绩效归因（Brinson 归因、因子暴露、风格分析）
"""

from linkbridge_finance import rates, risk, valuation, technical, portfolio, attribution
