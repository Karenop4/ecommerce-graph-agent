import os
from dataclasses import dataclass

from dotenv import load_dotenv


@dataclass(frozen=True)
class Settings:
    openai_api_key: str | None
    neo4j_uri: str | None
    neo4j_auth: tuple[str, str | None]
    redis_url: str
    redis_ttl_seconds: int
    mlflow_tracking_uri: str
    mlflow_experiment: str


def load_settings() -> Settings:
    load_dotenv(dotenv_path="../.env")
    load_dotenv(dotenv_path=".env")

    return Settings(
        openai_api_key=os.getenv("OPENAI_API_KEY"),
        neo4j_uri=os.getenv("NEO4J_URI"),
        neo4j_auth=(os.getenv("NEO4J_USER", "neo4j"), os.getenv("NEO4J_PASSWORD")),
        redis_url=os.getenv("REDIS_URL", "redis://localhost:6379/0"),
        redis_ttl_seconds=int(os.getenv("REDIS_TTL_SECONDS", "86400")),
        mlflow_tracking_uri=os.getenv("MLFLOW_TRACKING_URI") or "http://localhost:5000",
        mlflow_experiment="Agente_Produccion_Conversacional_Redis",
    )
