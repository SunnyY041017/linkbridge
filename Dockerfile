# LinkBridge Backend Dockerfile
# 多阶段构建 — 先安装项目包，再运行服务

FROM python:3.13-slim

WORKDIR /app

# 系统依赖
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc g++ \
    && rm -rf /var/lib/apt/lists/*

# 安装 Python 依赖
COPY backend/requirements.txt /app/backend/
RUN pip install --no-cache-dir -r /app/backend/requirements.txt

# 安装项目本地包
COPY linkbridge-core /app/linkbridge-core
COPY linkbridge-finance /app/linkbridge-finance
RUN pip install -e /app/linkbridge-finance -e /app/linkbridge-core

# 复制应用代码
COPY backend /app/backend

# 创建非 root 用户
RUN useradd -m -u 1000 linkbridge && chown -R linkbridge:linkbridge /app
USER linkbridge

WORKDIR /app/backend

EXPOSE 8000

CMD ["python", "run.py"]
