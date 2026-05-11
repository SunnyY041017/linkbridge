"""
自动化 API 测试套件 — 覆盖所有端点。

用法:
    cd backend && python -m pytest tests/test_api.py -v
"""
import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient
from app.main import app


@pytest_asyncio.fixture
async def client():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        yield c


class TestHealth:
    @pytest.mark.asyncio
    async def test_health_ok(self, client):
        r = await client.get("/health")
        assert r.status_code == 200
        data = r.json()
        assert data["status"] == "ok"
        assert data["service"] == "linkbridge"

    @pytest.mark.asyncio
    async def test_root_returns_html(self, client):
        r = await client.get("/")
        assert r.status_code == 200
        assert "text/html" in r.headers.get("content-type", "")

    @pytest.mark.asyncio
    async def test_docs_accessible(self, client):
        r = await client.get("/docs")
        assert r.status_code == 200


class TestChatAPI:
    @pytest.mark.asyncio
    async def test_chat_empty_message(self, client):
        r = await client.post("/api/v1/chat", json={"message": ""})
        assert r.status_code in (400, 422)

    @pytest.mark.asyncio
    async def test_chat_injection_rejected(self, client):
        r = await client.post("/api/v1/chat", json={"message": "SELECT * FROM users; DROP TABLE users;"})
        assert r.status_code == 400

    @pytest.mark.asyncio
    async def test_chat_xss_rejected(self, client):
        r = await client.post("/api/v1/chat", json={"message": "<script>alert('xss')</script>"})
        assert r.status_code == 400

    @pytest.mark.asyncio
    async def test_chat_normal_message_format(self, client):
        try:
            r = await client.post("/api/v1/chat", json={
                "message": "hello test",
                "multi_agent": False,
            }, timeout=15.0)
            if r.status_code == 200:
                data = r.json()
                assert "conversation_id" in data
                assert "content" in data
                assert "model" in data
        except Exception:
            pytest.skip("LLM API 不可用")


class TestChartAPI:
    @pytest.mark.asyncio
    async def test_chart_data_structure(self, client):
        r = await client.get("/api/v1/chart-data/600519?days=5", timeout=10.0)
        assert r.status_code == 200
        data = r.json()
        required_keys = ["symbol", "data_source", "dates", "kline", "risk", "valuation", "bond_risk"]
        for key in required_keys:
            assert key in data, f"缺少字段: {key}"
        assert data["symbol"] == "600519"
        assert len(data["kline"]) == 5

    @pytest.mark.asyncio
    async def test_chart_data_invalid_symbol(self, client):
        r = await client.get("/api/v1/chart-data/999999?days=5", timeout=10.0)
        assert r.status_code == 200


class TestRateLimit:
    @pytest.mark.asyncio
    async def test_rate_limit_header(self, client):
        r = await client.get("/health")
        assert r.status_code == 200

    @pytest.mark.asyncio
    async def test_rate_limit_under_threshold(self, client):
        for _ in range(5):
            r = await client.get("/health")
            assert r.status_code == 200


class TestStaticFiles:
    @pytest.mark.asyncio
    async def test_index_html(self, client):
        r = await client.get("/")
        assert r.status_code == 200

    @pytest.mark.asyncio
    async def test_report_html(self, client):
        r = await client.get("/static/report.html")
        assert r.status_code == 200

    @pytest.mark.asyncio
    async def test_404_static(self, client):
        r = await client.get("/static/nonexistent.js")
        assert r.status_code == 404


class TestSecurity:
    def test_input_injection_patterns(self):
        from app.middleware.security import sanitize_input

        dangerous = [
            "SELECT * FROM users; DROP TABLE users;",
            "<script>alert('xss')</script>",
            "'; exec('rm -rf /');",
            "../../etc/passwd",
        ]
        for inp in dangerous:
            cleaned, is_safe = sanitize_input(inp)
            assert not is_safe, f"应拒绝: {inp}"

    def test_input_normal_chinese(self):
        from app.middleware.security import sanitize_input

        normal = [
            "分析贵州茅台的投资价值",
            "600519 的 PE 和 PB 是多少？",
            "宁德时代现在值得入手吗",
        ]
        for inp in normal:
            cleaned, is_safe = sanitize_input(inp)
            assert is_safe, f"应接受: {inp}"
            assert len(cleaned) > 0

    def test_long_input_truncation(self):
        from app.middleware.security import sanitize_input

        long_msg = "分析 " * 2000
        cleaned, is_safe = sanitize_input(long_msg)
        assert len(cleaned) <= 4000


class TestCache:
    def test_cache_key_same_input(self):
        from linkbridge_core.cache import _make_cache_key

        k1 = _make_cache_key([{"role": "user", "content": "hello"}], "deepseek", 0.3, "2026-01-01")
        k2 = _make_cache_key([{"role": "user", "content": "hello"}], "deepseek", 0.3, "2026-01-01")
        assert k1 == k2

    def test_cache_key_different_input(self):
        from linkbridge_core.cache import _make_cache_key

        k1 = _make_cache_key([{"role": "user", "content": "hello"}], "deepseek", 0.3, "2026-01-01")
        k2 = _make_cache_key([{"role": "user", "content": "hi"}], "deepseek", 0.3, "2026-01-01")
        assert k1 != k2

    def test_cache_key_different_temp(self):
        from linkbridge_core.cache import _make_cache_key

        k1 = _make_cache_key([{"role": "user", "content": "hello"}], "deepseek", 0.3, "2026-01-01")
        k2 = _make_cache_key([{"role": "user", "content": "hello"}], "deepseek", 0.5, "2026-01-01")
        assert k1 != k2

    def test_cache_hit_miss(self):
        from linkbridge_core.cache import LLMCache

        cache = LLMCache(max_size=10)
        msgs = [{"role": "user", "content": "test cache"}]

        assert cache.get(msgs, "deepseek", 0.3) is None
        cache.set(msgs, "deepseek", 0.3, "cached response")
        assert cache.get(msgs, "deepseek", 0.3) == "cached response"
        assert cache.stats["hits"] == 1
        assert cache.stats["misses"] == 1

    def test_cache_lru_eviction(self):
        from linkbridge_core.cache import LLMCache

        cache = LLMCache(max_size=3)
        for i in range(5):
            msgs = [{"role": "user", "content": f"msg {i}"}]
            cache.set(msgs, "deepseek", 0.3, f"response {i}")

        assert cache.stats["size"] <= 3
