# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 项目概述

灵桥 LinkBridge — AI 原生投资研究智能体平台。面向个人投资者，用 Multi-Agent 协作将机构级投研能力带给普通用户。用户自然语言提问（如"分析宁德时代"），系统自动调度多个专业 Agent，从原始数据本地计算金融指标，再结合 LLM 推理生成结构化投研报告。

核心技术栈：Python 3.11+ / FastAPI / DeepSeek V4 / NumPy+SciPy / PostgreSQL+pgvector / Redis

## 架构三层的依赖关系

```
linkbridge-finance (纯计算库, 6模块, 71个单元测试)  ← 最底层, 可独立使用
    ↑
linkbridge-core   (Agent框架, Orchestrator, 依赖openai) ← 中间层, 依赖 finance
    ↑
backend/          (FastAPI服务, 6个Agent, 连线层)       ← 最上层, 依赖 core + finance
```

**linkbridge-finance** — 金融指标计算引擎，纯 Python + NumPy/SciPy/pandas。6 个模块：
- `rates`：久期/修正久期/凸性/DV01/有效久期
- `risk`：Sharpe/Sortino/Beta/Alpha/MaxDrawdown/VaR(三方法)/CVaR
- `valuation`：PE/PB/PS/PEG/DCF/Gordon/EV-EBITDA/NCAV
- `technical`：SMA/EMA/MACD/RSI/KDJ/Bollinger/ATR/OBV/CCI/WilliamsR
- `portfolio`：有效前沿/最小方差/最大夏普/风险平价/分散化比率
- `attribution`：Brinson归因/因子暴露/StyleAnalysis/IC分析

函数式设计，无状态，输入 DataFrame/array 输出 float。

**linkbridge-core** — Multi-Agent SDK。
- `BaseAgent`：Agent 生命周期（system_prompt → execute/execute_stream）
- `Orchestrator`：LLM 任务分解 → 依赖解析 → 并行调度 → 结果合成
- `LLMHub`：多模型路由（OpenAI-compatible 通用适配）
- `Tool/​ToolRegistry`：工具抽象与注册中心
- `message.py`：AgentMessage/TaskRequest/TaskResult 消息协议

**backend/** — FastAPI 应用。
- `app/data/`：AKShareProvider + AutoDataProvider 自动降级 + FallbackDataProvider
- `app/agents/`：6 个专业 Agent（Market/Funda/Techn/Quant/Risk/Senti）
- `app/api/v1/chat.py`：REST + WebSocket，支持单 Agent 和多 Agent 编排两种模式
- `app/api/v1/portfolio.py`：组合分析 API（待实现）

## 关键设计决策

- **LLM 不输出数字，只解读数字**：所有量化指标由 linkbridge-finance 在本地精确计算，LLM 接收已算好的指标值并负责自然语言解读。杜绝 LLM 幻觉编造财务数据。
- **自研 Agent 框架而非 LangChain**：`linkbridge-core` 从 0 构建。BaseAgent 约 60 行，LLMHub 约 120 行，Orchestrator 约 280 行——刻意轻量。
- **数据层自动降级**：`AutoDataProvider` 优先 AKShare 真实数据，网络不可用时自动切换 `FallbackDataProvider`（几何布朗运动生成模拟价格），始终可用。
- **固定随机种子**：FallbackDataProvider 使用 `RandomState(42)`，确保模拟数据可复现。
- **依赖感知并行调度**：Orchestrator 解析子任务依赖关系，无依赖的任务并行执行（asyncio.gather），有依赖的等前序完成后再执行。

## 常用命令

```bash
# 安装（在项目根目录）
pip install -e linkbridge-finance -e linkbridge-core
pip install -r backend/requirements.txt

# 运行全部测试（71 个）
python -m pytest linkbridge-finance/tests/ -v

# 运行单个模块测试
python -m pytest linkbridge-finance/tests/test_portfolio.py -v
python -m pytest linkbridge-finance/tests/test_attribution.py -v

# 单 Agent 测试
cd backend && python test_agent.py        # 结果 → test_result.txt
cd backend && python test_orchestrator.py  # 结果 → test_orchestrator_result.txt

# 全链路 E2E 测试（6 Agent + Orchestrator，耗时较长）
cd backend && python test_e2e.py           # 结果 → test_e2e_result.txt

# 启动开发服务器
cd backend && python run.py
# HTTP:  http://localhost:8000/docs  (Swagger UI)
# WS:    ws://localhost:8000/api/v1/ws/chat

# 测试单 Agent API
python -c "
import httpx, asyncio
async def t():
    r = await httpx.AsyncClient(timeout=60).post(
        'http://localhost:8000/api/v1/chat',
        json={'message': '分析600519', 'symbol': '600519'}
    )
    print(r.json()['content'])
asyncio.run(t())
"

# 测试多 Agent 编排 API
python -c "
import httpx, asyncio
async def t():
    r = await httpx.AsyncClient(timeout=120).post(
        'http://localhost:8000/api/v1/chat',
        json={'message': '全面分析贵州茅台', 'symbol': '600519', 'multi_agent': True}
    )
    print(r.json()['content'])
asyncio.run(t())
"
```

## 配置

编辑项目根目录的 `.env`，必填：`DEEPSEEK_API_KEY`。可选：`TONGYI_API_KEY`、`ZHIPU_API_KEY`。
配置类在 `backend/app/config.py`，从 `../.env` 读取（相对于 backend 目录）。

## 环境注意事项

- Windows GBK 终端无法打印 emoji/中文特殊字符。所有脚本输出请写入 UTF-8 文件而非 print。
- 公司网络可能限制 `push2his.eastmoney.com`（AKShare 的数据源），系统会自动降级为模拟数据，不影响 LLM 分析链路测试。
- DeepSeek API 兼容 OpenAI SDK，通过 `openai.AsyncOpenAI` 调用，base_url 指向 `https://api.deepseek.com/v1`。
- 清理残留进程：`cmd //c "taskkill /F /IM python.exe /T"` 从 bash，或直接在 PowerShell 运行 `taskkill /F /IM python.exe /T`。
