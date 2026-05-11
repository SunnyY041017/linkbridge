# 灵桥 LinkBridge — AI 原生投资研究智能体平台

面向个人投资者的 AI 投研助手。用户用自然语言提问（如"宁德时代现在值得入手吗？"），系统自动调度 6 个专业 AI Agent，从原始行情/财报数据出发，**在本地精确计算金融量化指标**，再结合大模型的推理能力，生成结构化投研报告。

> **核心设计：LLM 不输出数字，只解读数字** — 所有量化指标由自建金融引擎本地计算，彻底杜绝"AI 编造财务数据"的问题。

---

## 架构

```
微信小程序 / Web 前端 (ECharts)  ←→  FastAPI Gateway  ←→  Orchestrator
                                                              │
         ┌──────────┬──────────┬──────────┬──────────┬───────┤
         ▼          ▼          ▼          ▼          ▼       ▼
      Market     Funda      Techn      Quant      Risk    Senti
      Agent      Agent      Agent      Agent      Agent   Agent
         │          │          │          │          │       │
         └──────────┴──────────┴──────────┴──────────┴───────┘
                                │
                    linkbridge-finance (6 模块 · 71 测试)
                    夏普/久期/DCF/VaR/MACD/有效前沿/...
```

### 核心模块

| 模块 | 说明 | 状态 |
|------|------|------|
| `linkbridge-finance` | 金融指标计算引擎（纯 Python/NumPy/SciPy），6 模块 30+ 指标 | 71 测试 |
| `linkbridge-core` | Multi-Agent SDK（BaseAgent/Orchestrator/LLMHub/ToolRegistry） | 已发布 |
| `backend/` | FastAPI 应用（REST + WebSocket + 安全中间件 + 缓存） | 18 API 测试 |
| `static/` | Web 前端（SPA 对话界面 + ECharts 图表报告页） | 已上线 |

### 6 个专业 Agent

| Agent | 职责 | 核心指标 |
|-------|------|---------|
| **Market** | 行情分析 | Beta/Sharpe/波动率/VaR/Alpha |
| **Funda** | 基本面 | PE/PB/ROE/DCF估值/Gordon模型 |
| **Techn** | 技术面 | MACD/RSI/KDJ/布林带/均线/ATR/OBV/CCI |
| **Quant** | 量化分析 | 有效前沿/最小方差/最大夏普/风险平价 |
| **Risk** | 风险管理 | 久期/凸性/DV01/三方法VaR/压力测试 |
| **Senti** | 舆情情绪 | 市场情绪/舆论焦点/投资者心理 |

---

## 快速开始

### 环境要求
- Python 3.11+
- DeepSeek API Key（[免费注册](https://platform.deepseek.com)）

### 安装

```bash
git clone https://github.com/yourname/linkbridge.git
cd linkbridge

# 安装核心包
pip install -e linkbridge-finance -e linkbridge-core
pip install -r backend/requirements.txt

# 配置 API Key
cp .env.example .env
# 编辑 .env，填入 DEEPSEEK_API_KEY=sk-xxx
```

### 运行

```bash
# 启动开发服务器
cd backend && python run.py
# 访问: http://localhost:8000
# API 文档: http://localhost:8000/docs
# 研究报告: http://localhost:8000/static/report.html
```

### Docker 部署

```bash
# 设置环境变量
export DEEPSEEK_API_KEY=sk-xxx

# 启动全部服务
docker-compose up -d

# 查看日志
docker-compose logs -f app
```

---

## API

### 对话接口

```bash
# 单 Agent 模式
curl -X POST http://localhost:8000/api/v1/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "分析贵州茅台 600519"}'

# 多 Agent 编排模式
curl -X POST http://localhost:8000/api/v1/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "全面分析宁德时代", "multi_agent": true}'
```

### WebSocket 流式对话

```javascript
const ws = new WebSocket("ws://localhost:8000/api/v1/ws/chat");
ws.send(JSON.stringify({ message: "分析600519", multi_agent: true }));

ws.onmessage = (e) => {
  const data = JSON.parse(e.data);
  if (data.type === "chunk") console.log(data.content); // 流式输出
  if (data.type === "plan") console.log("编排:", data.content.agents);
  if (data.type === "agent_results") console.log("Agent 完成:", data.content);
};
```

### 图表数据

```bash
curl http://localhost:8000/api/v1/chart-data/600519?days=90
# 返回 K线/技术指标/风险指标/估值/债券数据 — 可直接用于 ECharts
```

---

## 测试

```bash
# 金融引擎单元测试（71 个）
python -m pytest linkbridge-finance/tests/ -v

# API 自动化测试（18 个）
cd backend && python -m pytest tests/test_api.py -v

# 端到端编排测试
cd backend && python test_e2e.py
# 结果 → test_e2e_result.txt

# Prompt 评测
cd backend && python scripts/eval_prompts.py --verbose
# 报告 → prompts/eval_report.md
```

---

## 项目结构

```
linkbridge/
├── linkbridge-finance/         # 金融指标计算引擎
│   ├── linkbridge_finance/
│   │   ├── rates.py            # 久期/修正久期/凸性/DV01
│   │   ├── risk.py             # Sharpe/Beta/Alpha/VaR/CVaR
│   │   ├── valuation.py        # PE/PB/DCF/Gordon/EV-EBITDA
│   │   ├── technical.py        # MACD/RSI/KDJ/布林带/均线
│   │   ├── portfolio.py        # 有效前沿/最小方差/风险平价
│   │   └── attribution.py      # Brinson归因/因子暴露/IC分析
│   └── tests/                  # 71 单元测试
│
├── linkbridge-core/            # Multi-Agent SDK
│   └── linkbridge_core/
│       ├── agent.py            # BaseAgent 抽象基类
│       ├── orchestrator.py     # Orchestrator 编排器
│       ├── llm_hub.py          # LLMHub 多模型路由 + 缓存
│       ├── tool.py             # Tool/ToolRegistry
│       ├── cache.py            # LLM 响应缓存（LRU + 持久化）
│       └── message.py          # 消息协议
│
├── backend/                    # FastAPI 应用
│   ├── app/
│   │   ├── api/v1/
│   │   │   ├── chat.py         # REST + WebSocket 对话端点
│   │   │   └── charts.py       # 图表数据 API
│   │   ├── agents/             # 6 个专业 Agent
│   │   ├── data/               # 数据层（AKShare + 自动降级）
│   │   ├── middleware/         # 安全中间件（限流 + 注入检测）
│   │   └── main.py             # 应用入口
│   ├── static/                 # Web 前端
│   │   ├── index.html          # 对话界面（流式 + Markdown）
│   │   └── report.html         # 报告页（6 张 ECharts）
│   ├── tests/                  # API 自动化测试
│   └── scripts/                # 评测脚本
│
├── docker-compose.yml          # Docker 编排
├── Dockerfile                  # 应用镜像
└── .env.example                # 环境变量模板
```

---

## 关键设计决策

### 自建金融计算引擎，而非依赖 LLM

大模型不适合精确数学计算。linkbridge-finance 在本地用 NumPy/SciPy 精确计算每一个指标，LLM 只负责自然语言解读。这样彻底杜绝了 LLM"编造"财务数据的可能。

### 自研 Agent 框架，而非 LangChain

`linkbridge-core` 从零构建，BaseAgent ~60 行，LLMHub ~120 行，Orchestrator ~280 行。刻意保持极简，专注于投研领域的需要。

### 数据层自动降级

`AutoDataProvider` 优先使用 AKShare 真实数据，网络不可用时自动切换为基于几何布朗运动的模拟数据，确保系统始终可用。

---

## 技术栈

Python 3.11+ / FastAPI / DeepSeek V4 / NumPy+SciPy+pandas / ECharts / PostgreSQL+pgvector / Redis / Docker

## 许可证

MIT
