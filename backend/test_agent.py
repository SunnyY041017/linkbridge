"""端到端测试：MarketAgent 全流程"""
import asyncio
import os
import sys

os.chdir(os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("DEEPSEEK_API_KEY", "sk-f8ed9af0a1cd47ec95ed7ef4ad512c2a")

from app.config import settings
from app.data.akshare_provider import AKShareProvider
from app.agents.market_agent import MarketAgent
from linkbridge_core.llm_hub import create_deepseek_client, ChatMessage
from linkbridge_core.message import TaskRequest


async def test_deepseek_connection(buf):
    buf.append("=" * 60)
    buf.append("Test 1: DeepSeek API direct connection")
    buf.append("=" * 60)
    client = create_deepseek_client(settings.deepseek_api_key)
    msgs = [ChatMessage(role="user", content="say hi in one sentence")]
    resp = await client.chat(msgs)
    buf.append(f"Model: {resp.model}")
    buf.append(f"Response: {resp.content[:200]}")
    buf.append(f"Token usage: {resp.usage}")
    buf.append("[PASS] DeepSeek V4 connected\n")
    return True


async def test_akshare_data(buf):
    buf.append("=" * 60)
    buf.append("Test 2: AKShare data fetch")
    buf.append("=" * 60)
    provider = AKShareProvider()
    import datetime
    end = datetime.date.today().strftime("%Y%m%d")
    start = (datetime.date.today() - datetime.timedelta(days=30)).strftime("%Y%m%d")
    try:
        df = await provider.get_stock_history("000001", start, end)
        buf.append(f"(000001) 30-day data: {len(df)} rows")
        if not df.empty:
            cols = [c for c in df.columns if c in ['日期','收盘','涨跌幅'] or 'date' in str(c).lower() or 'close' in str(c).lower()]
            buf.append(str(df.tail(3)))
        buf.append("[PASS] AKShare data fetch OK\n")
        return True
    except Exception as e:
        buf.append(f"[WARN] AKShare failed: {e}")
        buf.append("(continuing with simulated data later)\n")
        return False


async def test_indicator_computation(buf):
    buf.append("=" * 60)
    buf.append("Test 3: Financial indicator computation")
    buf.append("=" * 60)
    import pandas as pd
    import numpy as np
    from linkbridge_finance.risk import (
        sharpe_ratio, max_drawdown, annualized_return, annualized_volatility,
    )

    np.random.seed(42)
    n = 120
    prices = 10 + np.cumsum(np.random.normal(0.0005, 0.015, n))
    returns = pd.Series(prices).pct_change().dropna()

    sr = sharpe_ratio(returns)
    mdd = max_drawdown(returns)
    ann_vol = annualized_volatility(returns)
    ann_ret = annualized_return(returns)

    buf.append(f"  Annualized return: {ann_ret*100:.2f}%")
    buf.append(f"  Annualized volatility: {ann_vol*100:.2f}%")
    buf.append(f"  Sharpe ratio: {sr:.2f}")
    buf.append(f"  Max drawdown: {mdd*100:.2f}%")
    buf.append("[PASS] Indicator computation OK\n")
    return True


async def test_market_agent_full_flow(buf):
    buf.append("=" * 60)
    buf.append("Test 4: MarketAgent full analysis (data + indicators + LLM)")
    buf.append("=" * 60)

    client = create_deepseek_client(settings.deepseek_api_key)
    agent = MarketAgent(llm_client=client)

    task = TaskRequest(
        task_id="test-001",
        agent_name="market_agent",
        instruction="分析 000001 平安银行的近期走势和风险水平",
        context={"symbol": "000001", "days": 60},
    )

    buf.append("Starting analysis...\n")
    full_response = ""
    async for chunk in agent.run_stream(task):
        full_response += chunk

    buf.append("--- LLM Response ---")
    buf.append(full_response)
    buf.append("--- End ---\n")

    if len(full_response) > 100:
        buf.append("[PASS] MarketAgent full flow OK\n")
        return True
    else:
        buf.append("[WARN] LLM returned short content\n")
        return False


async def main():
    buf = []
    results = []

    try:
        results.append(await test_deepseek_connection(buf))
    except Exception as e:
        buf.append(f"[FAIL] DeepSeek connection: {e}\n")
        results.append(False)

    results.append(await test_akshare_data(buf))

    try:
        results.append(await test_indicator_computation(buf))
    except Exception as e:
        buf.append(f"[FAIL] Indicator computation: {e}\n")
        results.append(False)

    if results[0]:
        try:
            results.append(await test_market_agent_full_flow(buf))
        except Exception as e:
            buf.append(f"[FAIL] MarketAgent flow: {e}\n")
            import traceback
            buf.append(traceback.format_exc())
            results.append(False)

    buf.append("=" * 60)
    buf.append("TEST SUMMARY")
    buf.append("=" * 60)
    passed = sum(results)
    total = len(results)
    buf.append(f"Passed: {passed}/{total}")
    if passed == total:
        buf.append("ALL TESTS PASSED!")
    else:
        buf.append(f"WARNING: {total - passed} tests need attention")

    output = "\n".join(buf)
    with open("test_result.txt", "w", encoding="utf-8") as f:
        f.write(output)
    # Also print basic info to console (ASCII only)
    print(f"Tests done. {passed}/{total} passed. See test_result.txt for details.")


if __name__ == "__main__":
    asyncio.run(main())
