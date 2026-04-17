import os
from urllib.parse import urlparse


FIXED_LLM_MODEL = "llama-3.2-3b-instruct"
FIXED_LLM_ENDPOINT = (
    "https://llama-32-3b-instruct-rag.apps.mays-demo.sandbox3060.opentlc.com"
)
FIXED_EMBEDDING_MODEL = "granite-embedding-english-r2"
FIXED_EMBEDDING_ENDPOINT = (
    "https://granite-embedding-english-r2-rag.apps.mays-demo.sandbox3060.opentlc.com"
)

DEFAULT_REDIS_HOST = "10.131.0.60"
DEFAULT_REDIS_PORT = 11739
DEFAULT_REDIS_PASSWORD = "RfXEZmVr"
DEFAULT_OPENSHIFT_AI_API_KEY = "unused"


def normalize_openai_base_url(endpoint: str) -> str:
    endpoint = endpoint.rstrip("/")
    parsed = urlparse(endpoint)
    if not parsed.scheme or not parsed.netloc:
        raise ValueError(f"Invalid endpoint URL: {endpoint}")
    if parsed.path.endswith("/v1"):
        return endpoint
    return f"{endpoint}/v1"


def llm_model_name() -> str:
    return os.environ.get("LLM_MODEL", FIXED_LLM_MODEL)


def llm_base_url() -> str:
    endpoint = os.environ.get("LLM_ENDPOINT", FIXED_LLM_ENDPOINT)
    return normalize_openai_base_url(endpoint)


def embedding_model_name() -> str:
    return os.environ.get("EMBEDDING_MODEL", FIXED_EMBEDDING_MODEL)


def embedding_base_url() -> str:
    endpoint = os.environ.get("EMBEDDING_ENDPOINT", FIXED_EMBEDDING_ENDPOINT)
    return normalize_openai_base_url(endpoint)


def openshift_ai_api_key() -> str:
    return os.environ.get("OPENSHIFT_AI_API_KEY", DEFAULT_OPENSHIFT_AI_API_KEY)


def redis_host() -> str:
    return os.environ.get("REDIS_HOST", DEFAULT_REDIS_HOST)


def redis_port() -> int:
    return int(os.environ.get("REDIS_PORT", DEFAULT_REDIS_PORT))


def redis_password() -> str:
    return os.environ.get("REDIS_PASSWORD", DEFAULT_REDIS_PASSWORD)


def redis_url() -> str:
    return f"redis://:{redis_password()}@{redis_host()}:{redis_port()}"
