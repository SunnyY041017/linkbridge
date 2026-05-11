"""聊天 API 端点 — /api/v1/chat"""
import uuid

from fastapi import APIRouter, Depends, HTTPException, WebSocket, WebSocketDisconnect
from pydantic import BaseModel, Field

from app.agents.llm_setup import get_llm_hub
from app.middleware.security import sanitize_input
from app.agents.market_agent import MarketAgent
from app.agents.funda_agent import FundaAgent
from app.agents.techn_agent import TechnAgent
from app.agents.quant_agent import QuantAgent
from app.agents.risk_agent import RiskAgent
from app.agents.senti_agent import SentiAgent
from app.api.deps import get_current_user, get_data_provider

router = APIRouter(tags=["chat"])


class ChatRequest(BaseModel):
    message: str = Field(..., description="用户消息")
    symbol: str | None = Field(None, description="股票代码（可选）")
    conversation_id: str | None = Field(None, description="会话 ID")
    history: list[dict] = Field(default_factory=list)
    multi_agent: bool = Field(False, description="是否启用多 Agent 编排")


class ChatResponse(BaseModel):
    conversation_id: str
    content: str
    model: str = "deepseek-chat"


def _build_all_agents():
    """构建所有 Agent"""
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


@router.post("/chat", response_model=ChatResponse)
async def chat(
    req: ChatRequest,
    user: dict | None = Depends(get_current_user),
):
    # 输入安全检查
    clean_msg, is_safe = sanitize_input(req.message)
    if not is_safe:
        raise HTTPException(status_code=400, detail="输入包含不安全内容")
    req.message = clean_msg

    conv_id = req.conversation_id or str(uuid.uuid4())

    if req.multi_agent:
        # 多 Agent 编排模式
        from linkbridge_core.orchestrator import Orchestrator
        from linkbridge_core.message import TaskRequest

        hub = get_llm_hub()
        llm = hub.default_client
        agents = _build_all_agents()

        orch = Orchestrator(llm_client=llm, agents=agents, plan_with_llm=True)
        result = await orch.run(req.message, synthesize=True)

        return ChatResponse(
            conversation_id=conv_id,
            content=result["synthesis"] or "综合报告生成失败",
        )
    else:
        # 单 Agent 模式（向后兼容）
        agent = _build_all_agents()["market_agent"]
        from linkbridge_core.message import TaskRequest

        task = TaskRequest(
            task_id=str(uuid.uuid4()),
            agent_name=agent.name,
            instruction=req.message,
            context={"symbol": req.symbol} if req.symbol else {},
            tools=[],
        )

        result = await agent.run(task, history=req.history)
        if result.status.value == "error":
            raise HTTPException(status_code=500, detail=result.error)

        return ChatResponse(
            conversation_id=conv_id,
            content=result.content,
        )


@router.websocket("/ws/chat")
async def chat_websocket(websocket: WebSocket):
    await websocket.accept()
    hub = get_llm_hub()
    llm = hub.default_client
    provider = get_data_provider()
    agent = MarketAgent(llm_client=llm, data_provider=provider)

    try:
        while True:
            data = await websocket.receive_json()
            message = data.get("message", "")
            symbol = data.get("symbol")
            history = data.get("history", [])
            multi_agent = data.get("multi_agent", False)

            if not message:
                await websocket.send_json({"type": "error", "content": "消息不能为空"})
                continue

            clean_msg, is_safe = sanitize_input(message)
            if not is_safe:
                await websocket.send_json({"type": "error", "content": "输入包含不安全内容"})
                continue
            message = clean_msg

            from linkbridge_core.message import TaskRequest

            if multi_agent:
                # 多 Agent 编排模式下的流式输出
                await websocket.send_json({"type": "status", "content": "orchestrating"})
                agents = _build_all_agents()
                from linkbridge_core.orchestrator import Orchestrator

                orch = Orchestrator(llm_client=llm, agents=agents, plan_with_llm=True)

                # 先执行各 Agent（非流式），再流式合成
                plan = await orch.plan(message)
                await websocket.send_json({
                    "type": "plan",
                    "content": {
                        "reasoning": plan.reasoning,
                        "agents": [t.agent_name for t in plan.sub_tasks],
                    },
                })

                results = await orch.execute_plan(plan)
                all_content = ""
                for name, r in results.items():
                    if r.status.value == "done":
                        all_content += f"\n## {name}\n{r.content[:300]}...\n"

                await websocket.send_json({
                    "type": "agent_results",
                    "content": {name: r.status.value for name, r in results.items()},
                })

                # 合成报告
                synthesis = await orch.synthesize(message, results)
                await websocket.send_json({"type": "chunk", "content": synthesis})
                await websocket.send_json({"type": "done"})
            else:
                task = TaskRequest(
                    task_id=str(uuid.uuid4()),
                    agent_name=agent.name,
                    instruction=message,
                    context={"symbol": symbol} if symbol else {},
                )

                await websocket.send_json({"type": "status", "content": "thinking"})
                async for chunk in agent.run_stream(task, history=history):
                    await websocket.send_json({"type": "chunk", "content": chunk})
                await websocket.send_json({"type": "done"})

    except WebSocketDisconnect:
        pass
