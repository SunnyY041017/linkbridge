# LinkBridge 腾讯云部署指南

## 一、准备云服务器

### 1. 购买服务器
- 产品：**腾讯云轻量应用服务器**（比 CVM 便宜 30%，够用）
- 配置：2核2G / 40GB SSD / 4Mbps
- 系统：**Ubuntu 22.04 LTS**（不要选 Windows）
- 地域：北京或上海
- 时长：先买 1 个月测试（约 ¥58），确认后买 1 年优惠

购买地址：https://cloud.tencent.com/product/lighthouse

### 2. 购买域名 + ICP 备案（小程序必需）
- 域名注册：https://console.cloud.tencent.com/domain
- 注册一个 `.cn` 或 `.com` 域名（约 ¥30/年）
- **ICP 备案**（微信小程序强制要求）：
  - 进入腾讯云备案系统：https://console.cloud.tencent.com/beian
  - 个人备案：身份证正反面 + 人脸核验
  - 审核约 7-15 个工作日，**先提交备案再做其他事**

### 3. 免费 SSL 证书
备案通过后，在 https://console.cloud.tencent.com/ssl 申请免费 TrustAsia 证书，绑定你的域名。

---

## 二、服务器初始化（拿到服务器后运行）

```bash
# SSH 登录（用腾讯云控制台的重置密码先设密码）
ssh root@你的服务器IP

# 1. 更新系统 + 安装 Docker
apt update && apt upgrade -y
curl -fsSL https://get.docker.com | bash

# 2. 安装 Docker Compose
apt install docker-compose-plugin -y

# 3. 克隆项目
git clone https://github.com/SunnyY041017/linkbridge.git
cd linkbridge

# 4. 配置环境变量
cp .env.example .env
nano .env  # 填入真实的 DEEPSEEK_API_KEY 和 JWT_SECRET

# 5. 上传 SSL 证书
# 将腾讯云下载的证书文件放到 deploy/ssl/ 目录：
#   deploy/ssl/fullchain.pem  (Nginx 证书)
#   deploy/ssl/privkey.pem    (私钥)

# 6. 修改 nginx.conf 中的 server_name
nano deploy/nginx.conf
# 将 server_name _; 改成你的域名，如 server_name api.yourdomain.cn;

# 7. 启动全部服务
docker compose up -d --build

# 8. 验证
curl https://localhost/health     # 健康检查
curl https://localhost/api/v1/chat  # API 测试
```

---

## 三、验证清单

| 检查项 | 命令/操作 |
|--------|----------|
| 容器运行状态 | `docker compose ps` |
| 应用日志 | `docker compose logs -f app` |
| HTTPS 生效 | 浏览器打开 `https://你的域名` |
| WebSocket 可用 | 前端发一条消息，看 Agent 是否有回复 |
| API 文档 | `https://你的域名/docs` |

---

## 四、小程序后台配置

1. 登录 https://mp.weixin.qq.com
2. 开发管理 → 服务器域名 → 添加：
   - request合法域名：`https://你的域名`
   - socket合法域名：`wss://你的域名`
   - uploadFile合法域名：`https://你的域名`

---

## 五、日常维护

```bash
# 查看日志
docker compose logs -f app --tail=100

# 重启服务
docker compose restart app

# 更新代码
git pull && docker compose up -d --build

# 备份数据库
docker exec linkbridge-db pg_dump -U linkbridge linkbridge > backup_$(date +%Y%m%d).sql
```
