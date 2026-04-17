import os
from typing import Any, List, Optional

import gradio as gr
from datasets import Dataset
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_redis import RedisChatMessageHistory
from ragas import evaluate
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import answer_relevancy, faithfulness
from redisvl.extensions.llmcache import SemanticCache
from redisvl.extensions.router import SemanticRouter
from redisvl.utils.utils import create_ulid

from workbench.shared.cached_llm import CachedLLM
from workbench.shared.converters import str_to_bool
from workbench.shared.llm_utils import (
    embedding_base_url,
    embedding_model_name,
    llm_base_url,
    llm_model_name,
    openshift_ai_api_key,
    redis_host,
    redis_password,
    redis_port,
    redis_url,
)
from workbench.shared.logger import logger
from workbench.shared.pdf_manager import PDFManager, PDFMetadata

os.environ["TOKENIZERS_PARALLELISM"] = "false"

PRELOADED_PDF = os.environ.get("PRELOADED_PDF_PATH", "afh1.pdf")


class ChatApp:
    def __init__(self) -> None:
        self.session_id = None
        self.pdf_manager = None
        self.current_pdf_index = None
        self.startup_error = None

        self.redis_host = redis_host()
        self.redis_port = redis_port()
        self.redis_password = redis_password()
        self.redis_url = redis_url()

        self.openshift_ai_api_key = openshift_ai_api_key()
        self.llm_base_url = llm_base_url()
        self.embedding_base_url = embedding_base_url()
        self.selected_llm = llm_model_name()
        self.selected_embedding_model = embedding_model_name()

        self.credentials_set = True
        self.initialized = False

        self.chunk_size = int(os.environ.get("DEFAULT_CHUNK_SIZE", 500))
        self.chunking_technique = os.environ.get(
            "DEFAULT_CHUNKING_TECHNIQUE", "Recursive Character"
        )
        self.chat_history = None
        self.N = 0
        self.count = 0
        self.use_semantic_cache = str_to_bool(
            os.environ.get("DEFAULT_USE_SEMANTIC_CACHE")
        )
        self.top_k = int(os.environ.get("DEFAULT_TOP_K", 3))
        self.distance_threshold = float(
            os.environ.get("DEFAULT_DISTANCE_THRESHOLD", 0.30)
        )
        self.llm_temperature = float(os.environ.get("DEFAULT_LLM_TEMPERATURE", 0.7))
        self.use_chat_history = str_to_bool(os.environ.get("DEFAULT_USE_CHAT_HISTORY"))
        self.use_semantic_router = str_to_bool(
            os.environ.get("DEFAULT_USE_SEMANTIC_ROUTER")
        )
        self.use_ragas = str_to_bool(os.environ.get("DEFAULT_USE_RAGAS"))

        self.llm = None
        self.evalutor_llm = None
        self.cached_llm = None
        self.vector_store = None
        self.llmcache = None
        self.index_name = None
        self.semantic_router = None

    def initialize(self):
        try:
            self.initialize_components()
            self.startup_error = None
        except Exception as exc:
            self.startup_error = str(exc)
            self.initialized = False
            logger.exception("OpenShift AI workbench initialization failed: %s", exc)

    def initialize_components(self):
        self.pdf_manager = PDFManager(self.redis_url)
        embeddings = self.get_embedding_model()

        try:
            self.pdf_manager.ensure_preloaded_pdf(
                PRELOADED_PDF,
                embeddings,
                self.chunk_size,
                self.chunking_technique,
            )
            logger.info("Preloaded PDF ready: %s", PRELOADED_PDF)
        except FileNotFoundError:
            logger.info("No preloaded PDF found at %s", PRELOADED_PDF)
        except Exception as exc:
            logger.warning("Failed to preload PDF %s: %s", PRELOADED_PDF, exc)

        logger.info("Performing data reconciliation...")
        try:
            fixed, removed, orphaned = self.pdf_manager.reconcile_data()
            if fixed > 0 or removed > 0 or orphaned > 0:
                logger.info(
                    "Reconciliation summary - Fixed: %s, Removed: %s, Orphaned cleaned: %s",
                    fixed,
                    removed,
                    orphaned,
                )
            else:
                logger.info("Data reconciliation complete - no issues found")
        except Exception as exc:
            logger.warning("Data reconciliation failed: %s", exc)

        logger.info("Initializing semantic router")
        self.semantic_router = SemanticRouter.from_yaml(
            "workbench/router.yaml", redis_url=self.redis_url, overwrite=True
        )
        logger.info("Semantic router initialized")

        if self.use_chat_history:
            self.chat_history = RedisChatMessageHistory(
                session_id=self.session_id, redis_url=self.redis_url
            )
        else:
            self.chat_history = None

        self.update_llm()
        self.initialized = True

    def initialize_session(self):
        self.session_id = create_ulid()
        if self.use_chat_history:
            self.chat_history = RedisChatMessageHistory(
                session_id=self.session_id,
                redis_url=self.redis_url,
                index_name="chat_history",
            )
        else:
            self.chat_history = None

        return {"session_id": self.session_id, "chat_history": self.chat_history}

    def get_llm(self):
        return ChatOpenAI(
            model=self.selected_llm,
            api_key=self.openshift_ai_api_key,
            base_url=self.llm_base_url,
            temperature=self.llm_temperature,
            max_retries=2,
        )

    def get_embedding_model(self):
        logger.info(
            "Generating embeddings with model %s at %s",
            self.selected_embedding_model,
            self.embedding_base_url,
        )
        return OpenAIEmbeddings(
            model=self.selected_embedding_model,
            api_key=self.openshift_ai_api_key,
            base_url=self.embedding_base_url,
            tiktoken_enabled=False,
            check_embedding_ctx_length=False,
            max_retries=2,
        )

    def __call__(self, file: str, chunk_size: int, chunking_technique: str) -> Any:
        self.chunk_size = chunk_size
        self.chunking_technique = chunking_technique
        return self.process_pdf(file, chunk_size, chunking_technique)

    def build_chain(self, history: List[gr.ChatMessage]):
        retriever = self.vector_store.as_retriever(search_kwargs={"k": self.top_k})

        messages = [
            (
                "system",
                """You are a helpful AI assistant. Use the following pieces of
                    context to answer the user's question. If you don't know the
                answer, just say that you don't know, don't try to make up an
                    answer. Please be as detailed as possible with your
                    answers.""",
            ),
            ("system", "Context: {context}"),
        ]

        if self.use_chat_history:
            for msg in history:
                messages.append((msg["role"], msg["content"]))

        messages.append(("human", "{input}"))
        messages.append(
            (
                "system",
                "Provide a helpful and accurate answer based on the given context and question:",
            )
        )
        prompt = ChatPromptTemplate.from_messages(messages)

        combine_docs_chain = create_stuff_documents_chain(self.cached_llm, prompt)
        rag_chain = create_retrieval_chain(retriever, combine_docs_chain)

        return rag_chain

    def update_chat_history(
        self, history: List[gr.ChatMessage], use_chat_history: bool, session_state
    ):
        self.use_chat_history = use_chat_history

        if session_state is None:
            session_state = self.initialize_session()

        if self.use_chat_history:
            if (
                "chat_history" not in session_state
                or session_state["chat_history"] is None
            ):
                session_state["chat_history"] = RedisChatMessageHistory(
                    session_id=session_state.get("session_id", create_ulid()),
                    redis_url=self.redis_url,
                    index_name="chat_history",
                )
        else:
            if "chat_history" in session_state and session_state["chat_history"]:
                try:
                    session_state["chat_history"].clear()
                except Exception as exc:
                    logger.info("Error clearing chat history: %s", exc)
            session_state["chat_history"] = None

        history.clear()
        return history, session_state

    def get_chat_history(self):
        if self.chat_history and self.use_chat_history:
            messages = self.chat_history.messages
            formatted_history = []
            for msg in messages:
                if msg.type == "human":
                    formatted_history.append(f"Human: {msg.content}\n")
                elif msg.type == "ai":
                    formatted_history.append(f"AI: {msg.content}\n")
            return "\n".join(formatted_history)
        return "No chat history available."

    def update_semantic_router(self, use_semantic_router: bool):
        self.use_semantic_router = use_semantic_router

    def update_ragas(self, use_ragas: bool):
        self.use_ragas = use_ragas

    def update_llm(self):
        self.llm = self.get_llm()
        self.evalutor_llm = LangchainLLMWrapper(self.llm)

        if self.use_semantic_cache and self.llmcache is not None:
            self.cached_llm = CachedLLM(self.llm, self.llmcache)
        else:
            self.cached_llm = self.llm

    def update_temperature(self, new_temperature: float):
        self.llm_temperature = new_temperature
        self.update_llm()

    def update_top_k(self, new_top_k: int):
        self.top_k = new_top_k

    def make_semantic_cache(self) -> SemanticCache:
        semantic_cache_index_name = f"llmcache:{self.index_name}"
        return SemanticCache(
            name=semantic_cache_index_name,
            redis_url=self.redis_url,
            distance_threshold=self.distance_threshold,
        )

    def clear_semantic_cache(self):
        semantic_cache = self.make_semantic_cache()
        semantic_cache.clear()

    def update_semantic_cache(self, use_semantic_cache: bool):
        self.use_semantic_cache = use_semantic_cache
        if self.use_semantic_cache and self.index_name:
            self.llmcache = self.make_semantic_cache()
        else:
            self.llmcache = None

        self.update_llm()

    def update_distance_threshold(self, new_threshold: float):
        self.distance_threshold = new_threshold
        if self.index_name:
            self.llmcache = self.make_semantic_cache()
            self.update_llm()

    def get_last_cache_status(self) -> bool:
        if isinstance(self.cached_llm, CachedLLM):
            return self.cached_llm.get_last_cache_status()
        return False

    def evaluate_response(self, query, result):
        ds = Dataset.from_dict(
            {
                "question": [query],
                "answer": [result["answer"]],
                "contexts": [[c.page_content for c in result["context"]]],
            }
        )

        try:
            eval_results = evaluate(
                dataset=ds,
                metrics=[faithfulness, answer_relevancy],
                llm=self.evalutor_llm,
            )
            return eval_results
        except Exception as exc:
            logger.info("Error during RAGAS evaluation: %s", exc)
            return {}

    def process_pdf(
        self,
        file,
        chunk_size: int,
        chunking_technique: str,
        selected_embedding_model: Optional[str] = None,
    ) -> Any:
        try:
            embeddings = self.get_embedding_model()

            self.index_name = self.pdf_manager.process_pdf_complete(
                file, chunk_size, chunking_technique, embeddings
            )
            self.current_pdf_index = self.index_name
            if selected_embedding_model:
                self.selected_embedding_model = selected_embedding_model

            self.vector_store = self.pdf_manager.load_pdf_complete(
                self.index_name, embeddings
            )
            self.update_semantic_cache(self.use_semantic_cache)
        except Exception as exc:
            logger.error("Error during process_pdf: %s", exc)
            raise

    def load_pdf(self, index_name: str) -> bool:
        try:
            embeddings = self.get_embedding_model()
            self.vector_store = self.pdf_manager.load_pdf_complete(index_name, embeddings)

            self.index_name = index_name
            self.current_pdf_index = index_name
            metadata = self.pdf_manager.get_pdf_metadata(index_name)
            if metadata:
                self.chunk_size = metadata.chunk_size
                self.chunking_technique = metadata.chunking_technique

            self.update_semantic_cache(self.use_semantic_cache)
            return True
        except Exception as exc:
            logger.error("Failed to load PDF %s: %s", index_name, exc)
            return False

    def search_pdfs(self, query: str = "") -> List[PDFMetadata]:
        return self.pdf_manager.search_pdfs(query)

    def get_pdf_file(self, index_name: str) -> Optional[str]:
        return self.pdf_manager.get_pdf_file(index_name)


def generate_feedback(evaluation_scores):
    if not evaluation_scores:
        return "RAGAS evaluation failed."

    feedback = ["RAGAS Metrics:"]
    for metric, score in evaluation_scores.items():
        feedback.append(f"  - {metric}: {score:.4f}")
    return "\n".join(feedback)
