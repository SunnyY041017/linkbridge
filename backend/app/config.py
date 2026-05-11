from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # Database
    database_url: str = "postgresql+asyncpg://linkbridge:linkbridge_dev@localhost:5432/linkbridge"
    database_url_sync: str = "postgresql://linkbridge:linkbridge_dev@localhost:5432/linkbridge"

    # Redis
    redis_url: str = "redis://localhost:6379/0"

    # JWT
    jwt_secret: str = "change-me-to-a-random-string"
    jwt_algorithm: str = "HS256"
    jwt_expire_minutes: int = 1440

    # DeepSeek V4
    deepseek_api_key: str = ""
    deepseek_base_url: str = "https://api.deepseek.com/v1"

    # Optional LLM providers
    tongyi_api_key: str = ""
    zhipu_api_key: str = ""

    # App
    app_host: str = "0.0.0.0"
    app_port: int = 8000
    debug: bool = True

    @property
    def deepseek_enabled(self) -> bool:
        return bool(self.deepseek_api_key)

    model_config = {"env_file": "../.env", "extra": "ignore"}


settings = Settings()
