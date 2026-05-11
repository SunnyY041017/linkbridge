"""
Orchestrator 端到端测试 — 验证多 Agent 编排流程。
结果写入 test_orchestrator_result.txt（UTF-8）。
"""
import asyncio
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

from app.agents.llm_setup import get_llm_hub
from app.agents.market_agent import MarketAgent
from app.api.deps import get_data_provider
from linkbridge_core.orchestrator import Orchestrator, SubTask, OrchestrationPlan

buf = []


def log(msg):
    buf.append(msg)
    # 同时写入文件以防中途崩溃
    with open("test_orchestrator_result.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(buf))


async def test_orchestrator_basic():
    """测试 Orchestrator 基础功能：规划 + 执行 + 合成"""
    log("=" * 60)
    log("测试 1: Orchestrator 基础编排流程")
    log("=" * 60)

    hub = get_llm_hub()
    llm = hub.default_client
    provider = get_data_provider()

    market_agent = MarketAgent(llm_client=llm, data_provider=provider)

    agents = {"market_agent": market_agent}
    orch = Orchestrator(llm_client=llm, agents=agents, plan_with_llm=False)

    query = "分析贵州茅台（600519）当前行情"
    result = await orch.run(query, synthesize=False)  # 跳过 LLM 合成以节省 token

    plan = result["plan"]
    results = result["results"]

    log(f"\n查询: {query}")
    log(f"编排策略: {plan.reasoning}")
    log(f"子任务数: {len(plan.sub_tasks)}")
    for t in plan.sub_tasks:
        log(f"  - {t.agent_name}: {t.instruction[:60]}...")

    for name, r in results.items():
        status = "成功" if r.status.value == "done" else "失败"
        content_len = len(r.content) if r.content else 0
        log(f"\n[{name}] 状态: {status}, 输出长度: {content_len}")

        if r.status.value == "done":
            log(f"  内容预览: {r.content[:200]}...")
        else:
            log(f"  错误: {r.error}")

    log(f"\n测试 1 完成")


async def test_dependency_resolution():
    """测试依赖解析功能"""
    log("\n" + "=" * 60)
    log("测试 2: 依赖解析")
    log("=" * 60)

    hub = get_llm_hub()
    llm = hub.default_client
    orch = Orchestrator(llm_client=llm, agents={"m": None, "f": None, "t": None})

    plan = OrchestrationPlan(
        original_query="test",
        sub_tasks=[
            SubTask(agent_name="m", instruction="行情分析"),
            SubTask(agent_name="f", instruction="基本面分析", depends_on=["m"]),
            SubTask(agent_name="t", instruction="技术分析"),
        ],
    )

    groups = orch._resolve_dependencies(plan)
    log(f"依赖层级数: {len(groups)}")
    for i, g in enumerate(groups):
        names = [t.agent_name for t in g]
        log(f"  层级 {i}: {names}")

    assert len(groups) == 2, f"期望 2 层，实际 {len(groups)}"
    # 第一层应包含 m 和 t（无依赖，可并行）
    assert {t.agent_name for t in groups[0]} == {"m", "t"}
    # 第二层应包含 f（依赖 m）
    assert {t.agent_name for t in groups[1]} == {"f"}
    log("依赖解析验证通过")


async def test_llm_planning():
    """测试 LLM 任务规划"""
    log("\n" + "=" * 60)
    log("测试 3: LLM 任务规划（需要 DeepSeek API）")
    log("=" * 60)

    hub = get_llm_hub()
    llm = hub.default_client

    orch = Orchestrator(
        llm_client=llm,
        agents={
            "market_agent": None,
        },
        plan_with_llm=True,
    )

    try:
        plan = await orch.plan("分析宁德时代的投资价值")
        log(f"规划思路: {plan.reasoning}")
        log(f"子任务数: {len(plan.sub_tasks)}")
        for t in plan.sub_tasks:
            log(f"  - [{t.agent_name}] {t.instruction}")
            log(f"    依赖: {t.depends_on if t.depends_on else '无'}")
        log("LLM 规划测试完成")
    except Exception as e:
        log(f"LLM 规划失败（非致命）: {e}")


async def main():
    log("LinkBridge Orchestrator 集成测试")
    log(f"测试时间: {__import__('datetime').datetime.now().isoformat()}\n")

    await test_dependency_resolution()
    await test_llm_planning()
    await test_orchestrator_basic()

    log("\n" + "=" * 60)
    log("全部测试完成")
    log("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
