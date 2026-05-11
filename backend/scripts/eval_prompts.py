"""
Prompt 评测脚本 — 自动化评估 LLM 输出质量。

用法:
    cd backend && python scripts/eval_prompts.py
    cd backend && python scripts/eval_prompts.py --cases 5  # 只跑 5 个 case

输出:
    prompts/eval_report.md  (Markdown 评测报告)
"""
import argparse
import asyncio
import json
import os
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from app.agents.llm_setup import get_llm_hub


# ===== 评测用例 =====

@dataclass
class EvalCase:
    id: str
    category: str  # market / fundamental / technical / orchestration
    query: str
    symbol: str = ""
    expected_keywords: list[str] = field(default_factory=list)
    min_chars: int = 200
    description: str = ""


EVAL_CASES = [
    # 行情分析
    EvalCase(
        id="market_001", category="market",
        query="分析 600519 贵州茅台的当前行情",
        symbol="600519",
        expected_keywords=["行情概述", "风险收益", "指标", "波动"],
        description="标准行情分析请求",
    ),
    EvalCase(
        id="market_002", category="market",
        query="000001 平安银行的 Sharpe 比率和最大回撤是多少？",
        symbol="000001",
        expected_keywords=["Sharpe", "最大回撤", "风险"],
        description="特定指标查询",
    ),

    # 基本面分析
    EvalCase(
        id="funda_001", category="fundamental",
        query="分析 600519 的估值水平，PE/PB 是否合理？",
        symbol="600519",
        expected_keywords=["PE", "PB", "估值", "DCF"],
        description="估值分析",
    ),
    EvalCase(
        id="funda_002", category="fundamental",
        query="000858 五粮液的 ROE 和营收增速如何？",
        symbol="000858",
        expected_keywords=["ROE", "营收", "增长"],
        description="盈利能力分析",
    ),

    # 技术面
    EvalCase(
        id="techn_001", category="technical",
        query="300750 宁德时代的 MACD 和 RSI 指标信号是什么？",
        symbol="300750",
        expected_keywords=["MACD", "RSI", "技术"],
        description="技术指标分析",
    ),

    # 综合/编排
    EvalCase(
        id="orch_001", category="orchestration",
        query="全面分析 600519 贵州茅台，包括行情、基本面和技术面",
        symbol="600519",
        expected_keywords=["行情", "基本面", "技术"],
        min_chars=500,
        description="多维度综合分析",
    ),

    # 风险
    EvalCase(
        id="risk_001", category="risk",
        query="600036 招商银行的 VaR 和系统性风险 Beta 是多少？",
        symbol="600036",
        expected_keywords=["VaR", "Beta", "风险"],
        description="风险评估请求",
    ),
]


# ===== 评分函数 =====

def score_completeness(response: str, case: EvalCase) -> int:
    """完整性：是否 >= 最小字符数"""
    return 10 if len(response) >= case.min_chars else int(10 * len(response) / case.min_chars)


def score_keywords(response: str, case: EvalCase) -> int:
    """关键词覆盖：预期关键词是否出现"""
    if not case.expected_keywords:
        return 10
    hits = sum(1 for kw in case.expected_keywords if kw.lower() in response.lower())
    return int(10 * hits / len(case.expected_keywords))


def score_structure(response: str) -> int:
    """结构分：是否有 Markdown 标题、表格"""
    score = 0
    if "###" in response or "##" in response:
        score += 4
    if "|" in response and "---" in response:
        score += 3
    if len(response.split("\n\n")) >= 3:
        score += 3
    return min(score, 10)


def score_safety(response: str) -> int:
    """安全分：不包含投资建议措辞"""
    risky_phrases = ["建议买入", "建议卖出", "一定上涨", "一定下跌", "保证收益", "稳赚"]
    for phrase in risky_phrases:
        if phrase in response:
            return 0
    return 10


def evaluate_response(response: str, case: EvalCase) -> dict:
    return {
        "completeness": score_completeness(response, case),
        "keywords": score_keywords(response, case),
        "structure": score_structure(response),
        "safety": score_safety(response),
        "total": 0,
    }


async def run_evaluation(max_cases: int = 0, verbose: bool = False) -> list[dict]:
    """运行评测"""
    hub = get_llm_hub()
    llm = hub.default_client
    from app.agents.market_agent import MarketAgent
    from app.agents.funda_agent import FundaAgent
    from app.agents.techn_agent import TechnAgent
    from app.agents.risk_agent import RiskAgent
    from app.agents.quant_agent import QuantAgent
    from app.agents.senti_agent import SentiAgent
    from app.api.deps import get_data_provider
    from linkbridge_core.orchestrator import Orchestrator
    from linkbridge_core.message import TaskRequest

    provider = get_data_provider()
    agents = {
        "market_agent": MarketAgent(llm_client=llm, data_provider=provider),
        "funda_agent": FundaAgent(llm_client=llm, data_provider=provider),
        "techn_agent": TechnAgent(llm_client=llm, data_provider=provider),
        "quant_agent": QuantAgent(llm_client=llm, data_provider=provider),
        "risk_agent": RiskAgent(llm_client=llm, data_provider=provider),
        "senti_agent": SentiAgent(llm_client=llm),
    }

    cases = EVAL_CASES[:max_cases] if max_cases > 0 else EVAL_CASES
    results = []
    agent_map = {
        "market": "market_agent",
        "fundamental": "funda_agent",
        "technical": "techn_agent",
        "risk": "risk_agent",
        "orchestration": "market_agent",
    }

    for i, case in enumerate(cases):
        if verbose:
            print(f"[{i+1}/{len(cases)}] {case.id}: {case.description}")

        start = time.time()

        try:
            if case.category == "orchestration" and case.symbol:
                orch = Orchestrator(llm_client=llm, agents=agents, plan_with_llm=False)
                result = await orch.run(case.query, synthesize=True)
                response = result.get("synthesis", "")
            else:
                agent_name = agent_map.get(case.category, "market_agent")
                agent = agents[agent_name]
                task = TaskRequest(
                    task_id=case.id,
                    agent_name=agent.name,
                    instruction=case.query,
                    context={"symbol": case.symbol} if case.symbol else {},
                )
                result = await agent.run(task)
                response = result.content

            elapsed = round(time.time() - start, 1)
            scores = evaluate_response(response, case)
            scores["total"] = sum(scores.values())
            scores["max"] = 40

            results.append({
                "case_id": case.id,
                "category": case.category,
                "description": case.description,
                "response_length": len(response),
                "elapsed_sec": elapsed,
                "scores": scores,
                "response_preview": response[:300] + "...",
            })

            if verbose:
                print(f"    Score: {scores['total']}/{scores['max']} ({elapsed}s, {len(response)} chars)")

        except Exception as e:
            elapsed = round(time.time() - start, 1)
            results.append({
                "case_id": case.id,
                "category": case.category,
                "description": case.description,
                "response_length": 0,
                "elapsed_sec": elapsed,
                "scores": {"completeness": 0, "keywords": 0, "structure": 0, "safety": 0, "total": 0, "max": 40},
                "error": str(e),
            })
            if verbose:
                print(f"    ERROR: {e}")

    return results


def generate_report(results: list[dict], output_path: str):
    """生成 Markdown 评测报告"""
    lines = []
    lines.append("# LinkBridge Prompt 评测报告\n")
    lines.append(f"> 评测时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"> 评测用例数: {len(results)}")
    lines.append("")

    # 总览
    passed = sum(1 for r in results if r.get("scores", {}).get("total", 0) >= 20)
    avg_score = sum(r.get("scores", {}).get("total", 0) for r in results) / max(len(results), 1)
    avg_time = sum(r.get("elapsed_sec", 0) for r in results) / max(len(results), 1)

    lines.append("## 总览\n")
    lines.append(f"| 指标 | 值 |")
    lines.append(f"|------|----|")
    lines.append(f"| 通过率 | {passed}/{len(results)} ({round(passed/len(results)*100, 1)}%) |")
    lines.append(f"| 平均分 | {round(avg_score, 1)}/40 |")
    lines.append(f"| 平均耗时 | {round(avg_time, 1)}s |")
    lines.append("")

    # 分类统计
    lines.append("## 分类得分\n")
    lines.append(f"| 类别 | 用例数 | 平均分 | 平均长度 | 平均耗时 |")
    lines.append(f"|------|--------|--------|----------|----------|")

    from collections import defaultdict
    cat_stats = defaultdict(lambda: {"count": 0, "total_score": 0, "total_len": 0, "total_time": 0})
    for r in results:
        cat = r["category"]
        cat_stats[cat]["count"] += 1
        cat_stats[cat]["total_score"] += r.get("scores", {}).get("total", 0)
        cat_stats[cat]["total_len"] += r.get("response_length", 0)
        cat_stats[cat]["total_time"] += r.get("elapsed_sec", 0)

    for cat, s in sorted(cat_stats.items()):
        n = s["count"]
        lines.append(
            f"| {cat} | {n} | {round(s['total_score']/n, 1)} | {round(s['total_len']/n)} | {round(s['total_time']/n, 1)}s |"
        )
    lines.append("")

    # 逐用例详情
    lines.append("## 逐用例详情\n")
    for r in results:
        scores = r.get("scores", {})
        lines.append(f"### {r['case_id']} — {r['description']}\n")
        lines.append(f"- **类别**: {r['category']}")
        lines.append(f"- **状态**: {'通过' if scores.get('total', 0) >= 20 else '失败'}")
        lines.append(f"- **得分**: {scores.get('total', 0)}/{scores.get('max', 40)}")
        lines.append(f"  - 完整性: {scores.get('completeness', 0)}/10")
        lines.append(f"  - 关键词: {scores.get('keywords', 0)}/10")
        lines.append(f"  - 结构化: {scores.get('structure', 0)}/10")
        lines.append(f"  - 安全性: {scores.get('safety', 0)}/10")
        lines.append(f"- **耗时**: {r.get('elapsed_sec', 0)}s")
        lines.append(f"- **响应长度**: {r.get('response_length', 0)} 字符")

        if "error" in r:
            lines.append(f"- **错误**: {r['error']}")
        else:
            preview = r.get("response_preview", "")[:200]
            lines.append(f"- **预览**: {preview}")
        lines.append("")

    report = "\n".join(lines)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(report)

    return report


async def main():
    parser = argparse.ArgumentParser(description="LinkBridge Prompt 评测")
    parser.add_argument("--cases", type=int, default=0, help="最大评测用例数（0=全部）")
    parser.add_argument("--output", type=str, default="prompts/eval_report.md", help="输出路径")
    parser.add_argument("--verbose", "-v", action="store_true", help="详细输出")
    args = parser.parse_args()

    print(f"LinkBridge Prompt 评测 ({datetime.now().strftime('%H:%M')})")
    print(f"用例数: {args.cases or len(EVAL_CASES)}")
    print()

    results = await run_evaluation(max_cases=args.cases, verbose=args.verbose)
    report = generate_report(results, args.output)

    print(f"\n报告已生成: {args.output}")
    avg = sum(r.get("scores", {}).get("total", 0) for r in results) / max(len(results), 1)
    print(f"平均分: {round(avg, 1)}/40")


if __name__ == "__main__":
    asyncio.run(main())
