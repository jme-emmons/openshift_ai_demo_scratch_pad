import os
from urllib.parse import urlparse


DEFAULT_LLM_MODEL = "llama-3.2-3b-instruct"
DEFAULT_LLM_ENDPOINT = ""
DEFAULT_EMBEDDING_MODEL = "granite-embedding-english-r2"
DEFAULT_EMBEDDING_ENDPOINT = ""

DEFAULT_REDIS_HOST = ""
DEFAULT_REDIS_PORT = 6379
DEFAULT_REDIS_PASSWORD = ""
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
    return os.environ.get("LLM_MODEL", DEFAULT_LLM_MODEL)


def llm_base_url() -> str:
    endpoint = os.environ.get("LLM_ENDPOINT", DEFAULT_LLM_ENDPOINT)
    if not endpoint:
        raise ValueError("LLM_ENDPOINT must be set")
    return normalize_openai_base_url(endpoint)


def embedding_model_name() -> str:
    return os.environ.get("EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL)


def embedding_base_url() -> str:
    endpoint = os.environ.get("EMBEDDING_ENDPOINT", DEFAULT_EMBEDDING_ENDPOINT)
    if not endpoint:
        raise ValueError("EMBEDDING_ENDPOINT must be set")
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
    host = redis_host()
    if not host:
        raise ValueError("REDIS_HOST must be set")
    password = redis_password()
    auth = f":{password}@" if password else ""
    return f"redis://{auth}{host}:{redis_port()}"
