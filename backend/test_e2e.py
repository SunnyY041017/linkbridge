"""
全链路端到端测试 — Orchestrator + 6 Agent 协同工作。
结果写入 test_e2e_result.txt（UTF-8）。
"""
import asyncio
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

from app.agents.llm_setup import get_llm_hub
from app.agents.market_agent import MarketAgent
from app.agents.funda_agent import FundaAgent
from app.agents.techn_agent import TechnAgent
from app.agents.quant_agent import QuantAgent
from app.agents.risk_agent import RiskAgent
from app.agents.senti_agent import SentiAgent
from app.api.deps import get_data_provider
from linkbridge_core.orchestrator import Orchestrator

buf = []


def log(msg):
    buf.append(msg)
    with open("test_e2e_result.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(buf))


def build_agents():
    hub = get_llm_hub()
    llm = hub.default_client
    provider = get_data_provider()

    return {
        "market_agent": MarketAgent(llm_client=llm, data_provider=provider),
        "funda_agent": FundaAgent(llm_client=llm, data_provider=provider),
        "techn_agent": TechnAgent(llm_client=llm, data_provider=provider),
        "quant_agent": QuantAgent(llm_client=llm, data_provider=provider),
        "risk_agent": RiskAgent(llm_client=llm, data_provider=provider),
        "senti_agent": SentiAgent(llm_client=llm),
    }


async def test_orchestrator_multi_agent():
    """测试 Orchestrator 调度 6 个 Agent"""
    log("=" * 60)
    log("测试 1: Orchestrator 六 Agent 并行调度")
    log("=" * 60)

    hub = get_llm_hub()
    llm = hub.default_client
    agents = build_agents()

    orch = Orchestrator(
        llm_client=llm,
        agents=agents,
        plan_with_llm=False,  # 使用默认全并行策略
        agent_timeout=60.0,
    )

    query = "全面分析贵州茅台（600519），包括行情、基本面、技术面、量化风险和舆情"
    log(f"\n查询: {query}")
    log(f"注册 Agent 数: {len(agents)}")
    for name in agents:
        log(f"  - {name}")

    result = await orch.run(query, synthesize=False)

    plan = result["plan"]
    results = result["results"]

    log(f"\n编排策略: {plan.reasoning}")
    log(f"子任务数: {len(plan.sub_tasks)}")
    log(f"执行结果数: {len(results)}")

    success_count = 0
    for name, r in results.items():
        if r.status.value == "done":
            success_count += 1
            content_preview = r.content[:150].replace("\n", " ")
            log(f"  [{name}] 成功 — 输出 {len(r.content)} 字符")
            log(f"    预览: {content_preview}...")
        else:
            log(f"  [{name}] 失败 — {r.error}")

    log(f"\n成功: {success_count}/{len(results)}")
    log("测试 1 完成")


async def test_synthesis():
    """测试 LLM 综合报告生成"""
    log("\n" + "=" * 60)
    log("测试 2: LLM 综合报告合成")
    log("=" * 60)

    hub = get_llm_hub()
    llm = hub.default_client
    agents = build_agents()

    # 只用 2 个 Agent 做快速合成测试
    orch = Orchestrator(
        llm_client=llm,
        agents={
            "market_agent": agents["market_agent"],
            "funda_agent": agents["funda_agent"],
        },
        plan_with_llm=False,
    )

    query = "分析贵州茅台（600519）的投资价值"
    log(f"\n查询: {query}")

    result = await orch.run(query, synthesize=True)
    synthesis = result.get("synthesis", "")

    log(f"\n合成报告长度: {len(synthesis)} 字符")
    log(f"合成报告预览:\n{synthesis[:500]}...")

    log("\n测试 2 完成")


async def test_dependency_chain():
    """测试依赖链：市场数据 → 技术分析（依赖市场数据）"""
    log("\n" + "=" * 60)
    log("测试 3: 依赖链执行")
    log("=" * 60)

    hub = get_llm_hub()
    llm = hub.default_client
    agents = build_agents()

    orch = Orchestrator(llm_client=llm, agents=agents, plan_with_llm=False)

    # 手动构造带依赖的计划
    from linkbridge_core.orchestrator import OrchestrationPlan, SubTask

    plan = OrchestrationPlan(
        original_query="分析 600519",
        sub_tasks=[
            SubTask(
                agent_name="market_agent",
                instruction="获取 600519 行情数据并计算风险收益指标",
                depends_on=[],
            ),
            SubTask(
                agent_name="techn_agent",
                instruction="基于行情数据做 600519 的技术分析",
                depends_on=["market_agent"],
            ),
            SubTask(
                agent_name="funda_agent",
                instruction="分析 600519 的基本面",
                depends_on=[],
            ),
        ],
    )

    log(f"\n计划: {len(plan.sub_tasks)} 个子任务")
    for t in plan.sub_tasks:
        deps = t.depends_on if t.depends_on else "无"
        log(f"  - [{t.agent_name}] 依赖: {deps}")

    results = await orch.execute_plan(plan)

    for name, r in results.items():
        status = "成功" if r.status.value == "done" else "失败"
        log(f"\n[{name}] {status}")
        if r.status.value == "done":
            log(f"  输出长度: {len(r.content)} 字符")

    log("\n测试 3 完成")


async def main():
    log("LinkBridge 全链路端到端测试")
    log(f"测试时间: {__import__('datetime').datetime.now().isoformat()}")
    log(f"Agent 总数: 6 (Market + Funda + Techn + Quant + Risk + Senti)")
    log("")

    await test_dependency_chain()
    await test_orchestrator_multi_agent()
    await test_synthesis()

    log("\n" + "=" * 60)
    log("全部 E2E 测试完成")
    log("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
