import os
import time
from datetime import datetime
from typing import List

import gradio as gr
from dotenv import load_dotenv
from gradio_modal import Modal
from gradio_pdf import PDF

from workbench.chat_app import ChatApp, generate_feedback
from workbench.shared.converters import str_to_bool
from workbench.shared.llm_utils import (
    embedding_base_url,
    embedding_model_name,
    llm_base_url,
    llm_model_name,
)
from workbench.shared.logger import logger
from workbench.shared.theme_management import load_theme

load_dotenv()

app = ChatApp()
redis_theme, redis_styles = load_theme("redis")


def path():
    return "/workbench"


def app_title():
    return "Chat with one PDF"


TAG_ESCAPE_CHARS = {
    ",",
    ".",
    "<",
    ">",
    "{",
    "}",
    "[",
    "]",
    '"',
    "'",
    ":",
    ";",
    "!",
    "@",
    "#",
    "$",
    "%",
    "^",
    "&",
    "*",
    "(",
    ")",
    "-",
    "+",
    "=",
    "~",
    "|",
    "/",
}


def escape_redis_search_query(query: str) -> str:
    return "".join(f"\\{char}" if char in TAG_ESCAPE_CHARS else char for char in query)


def add_text(history: List[gr.ChatMessage], text: str):
    if not text:
        raise gr.Error("enter text")

    if not app.use_chat_history:
        history.clear()

    history.append(gr.ChatMessage(role="user", content=text))
    return history


def initialize():
    app.initialize()


def configuration_summary() -> str:
    return (
        f"LLM: `{llm_model_name()}`\n\n"
        f"LLM endpoint: `{llm_base_url()}`\n\n"
        f"Embedding model: `{embedding_model_name()}`\n\n"
        f"Embedding endpoint: `{embedding_base_url()}`\n\n"
        f"Redis: `{app.redis_host}:{app.redis_port}`"
    )


def startup_notice() -> str:
    if app.startup_error:
        return (
            "Startup check could not fully initialize the backend.\n\n"
            f"Error: `{app.startup_error}`"
        )
    return "Backend initialized with the fixed OpenShift AI and Redis configuration."


def ensure_app_ready():
    if app.initialized:
        return
    if app.startup_error:
        raise gr.Error(f"Backend initialization failed: {app.startup_error}")
    raise gr.Error("Backend is not initialized yet.")


def reset_app():
    if app.index_name:
        app.clear_semantic_cache()

    app.chat_history = []
    app.N = 0
    app.current_pdf_index = None
    app.index_name = None
    app.chunk_size = int(os.environ.get("DEFAULT_CHUNK_SIZE", 500))
    app.chunking_technique = os.environ.get(
        "DEFAULT_CHUNKING_TECHNIQUE", "Recursive Character"
    )
    app.chat_history = None
    app.N = 0
    app.count = 0
    app.use_semantic_cache = str_to_bool(os.environ.get("DEFAULT_USE_SEMANTIC_CACHE"))
    app.top_k = int(os.environ.get("DEFAULT_TOP_K", 3))
    app.distance_threshold = float(os.environ.get("DEFAULT_DISTANCE_THRESHOLD", 0.30))
    app.llm_temperature = float(os.environ.get("DEFAULT_LLM_TEMPERATURE", 0.7))
    app.use_chat_history = str_to_bool(os.environ.get("DEFAULT_USE_CHAT_HISTORY"))
    app.use_semantic_router = str_to_bool(os.environ.get("DEFAULT_USE_SEMANTIC_ROUTER"))
    app.use_ragas = str_to_bool(os.environ.get("DEFAULT_USE_RAGAS"))

    return (
        [],
        None,
        "",
        gr.update(value=app.chunk_size),
        gr.update(value=app.chunking_technique),
        gr.update(value=app.use_semantic_cache),
        gr.update(value=app.top_k),
        gr.update(value=app.distance_threshold),
        gr.update(value=app.llm_temperature),
        gr.update(value=app.use_chat_history),
        gr.update(value=app.use_semantic_router),
        gr.update(value=app.use_ragas),
    )


def show_history(session_state):
    if app.use_chat_history and session_state["chat_history"] is not None:
        try:
            messages = session_state["chat_history"].messages
            formatted_history = []
            for msg in messages:
                if msg.type == "human":
                    formatted_history.append(f"Human: {msg.content}\n")
                elif msg.type == "ai":
                    formatted_history.append(f"AI: {msg.content}\n")
            history = "\n".join(formatted_history)
        except Exception as exc:
            logger.info("Error retrieving chat history: %s", exc)
            history = "Error retrieving chat history."
    else:
        history = "No chat history available."

    return history, gr.update(visible=True)


def render_first(file, chunk_size, chunking_technique, session_state):
    ensure_app_ready()
    if not session_state:
        session_state = app.initialize_session()

    app.process_pdf(file, chunk_size, chunking_technique)
    pdf_viewer = PDF(value=file.name, starting_page=1)

    return pdf_viewer, [], session_state


def perform_ragas_evaluation(query, result):
    evaluation_scores = app.evaluate_response(query, result)
    feedback = generate_feedback(evaluation_scores)
    return feedback


def get_response(
    history,
    query,
    file,
    distance_threshold,
    top_k,
    llm_temperature,
    session_state,
):
    ensure_app_ready()
    if not session_state:
        app.session_state = app.initialize_session()
        session_state = app.session_state
    if not file and not app.current_pdf_index:
        raise gr.Error(message="Please upload or select a PDF first")

    if app.top_k != top_k:
        app.update_top_k(top_k)
    if app.distance_threshold != distance_threshold:
        app.update_distance_threshold(distance_threshold)
    if app.llm_temperature != llm_temperature:
        app.update_temperature(llm_temperature)

    chain = app.build_chain(history)
    start_time = time.time()
    result = chain.invoke({"input": query})
    route = app.semantic_router(query) if app.use_semantic_router else None
    elapsed_time = time.time() - start_time
    is_cache_hit = app.get_last_cache_status()

    logger.info("Cache Hit: %s", is_cache_hit)
    if route:
        logger.info("Semantic Router Hit: %s", route)

    if app.use_chat_history and session_state["chat_history"] is not None:
        session_state["chat_history"].add_user_message(query)
        session_state["chat_history"].add_ai_message(result["answer"])
    else:
        logger.info("Chat history not updated (disabled or None)")

    output = (
        f"Cache Hit: {elapsed_time:.2f} sec\n\n"
        if is_cache_hit
        else f"LLM: {elapsed_time:.2f} sec\n\n"
    )
    if app.use_semantic_router and route and route.name is not None:
        output += f"Router: {route}\n\n"

    history.append(gr.ChatMessage(role="assistant", content=result["answer"]))
    yield history, "", output, session_state

    if app.use_ragas:
        feedback = perform_ragas_evaluation(query, result)
        final_output = f"{output}\n\n{feedback}"
        yield history, "", final_output, session_state


def format_pdf_list(pdfs):
    return [
        [
            pdf.filename,
            pdf.file_size,
            datetime.fromisoformat(pdf.upload_date).strftime("%Y-%m-%d %H:%M"),
        ]
        for pdf in pdfs
    ]


def update_pdf_list(search_query=""):
    ensure_app_ready()
    pdfs = app.search_pdfs(f"*{search_query}*" if search_query else "*")
    return format_pdf_list(pdfs)


def show_pdf_selector():
    ensure_app_ready()
    return gr.update(visible=True), format_pdf_list(app.search_pdfs())


def handle_pdf_selection(evt: gr.SelectData, pdf_list):
    ensure_app_ready()
    try:
        selected_row = pdf_list.iloc[evt.index[0]]
        filename = selected_row.iloc[0]

        filename = escape_redis_search_query(filename)
        pdfs = app.search_pdfs("@filename:{" + filename + "}")
        if not pdfs:
            return None, [], "PDF not found", gr.update(visible=False)

        pdf = pdfs[0]
        success = app.load_pdf(pdf.index_name)
        if not success:
            return None, [], "Failed to load PDF", gr.update(visible=False)

        pdf_path = app.get_pdf_file(pdf.index_name)
        if not pdf_path:
            return None, [], "PDF file not found", gr.update(visible=False)

        pdf_viewer = PDF(value=pdf_path, starting_page=1)
        return pdf_viewer, [], f"Loaded {filename}", gr.update(visible=False)
    except Exception as exc:
        logger.info("Failed to handle PDF selection: %s", exc)
        return None, [], f"Error loading PDF: {exc}", gr.update(visible=False)


HEADER = """
<div style="display: flex; justify-content: center;">
    <img src="../assets/redis-logo.svg" style="height: 2rem">
</div>
<div style="text-align: center">
    <h1>RAG workbench</h1>
</div>
"""

CONFIG_MARKDOWN = """
### Fixed Runtime Configuration
This deployment is pinned to a single OpenShift AI LLM, a single OpenShift AI embedding model, and a single Redis instance.
"""

js_func = """
function refresh() {
    const url = new URL(window.location);

    if (url.searchParams.get('__theme') !== 'dark') {
        url.searchParams.set('__theme', 'dark');
        window.location.href = url.href;
    }
}
"""


def ui():
    with gr.Blocks(
        theme=redis_theme, css=redis_styles, title="RAG workbench", js=js_func
    ) as blocks:
        session_state = gr.State()
        hidden_file = gr.State(None)

        with gr.Row():
            gr.HTML(HEADER)

        with gr.Row():
            with gr.Column(scale=6):
                chatbot = gr.Chatbot(value=[], type="messages", elem_id="chatbot")
                feedback_markdown = gr.Markdown(
                    value=startup_notice(), label="Status", visible=True
                )

                with gr.Row(elem_id="input-row"):
                    txt = gr.Textbox(
                        show_label=False,
                        placeholder="Enter text and press enter",
                        elem_id="txt",
                        scale=4,
                    )
                    submit_btn = gr.Button(
                        "Submit", elem_classes="chat-button", scale=1
                    )
                    show_history_btn = gr.Button(
                        "Chat History", elem_classes="chat-button", scale=2
                    )

                with gr.Row(equal_height=True):
                    with gr.Column(min_width=210):
                        use_ragas = gr.Checkbox(label="Use RAGAS", value=app.use_ragas)
                    with gr.Column(min_width=210):
                        use_semantic_router = gr.Checkbox(
                            label="Use Semantic Router", value=app.use_semantic_router
                        )
                    with gr.Column(min_width=210):
                        use_chat_history = gr.Checkbox(
                            label="Use Chat History", value=app.use_chat_history
                        )

                with gr.Accordion(label="Cache Settings", elem_classes=["accordion"]):
                    with gr.Row():
                        use_semantic_cache = gr.Checkbox(
                            label="Use LangCache", value=app.use_semantic_cache
                        )
                        distance_threshold = gr.Slider(
                            minimum=0.01,
                            maximum=1.0,
                            value=app.distance_threshold,
                            step=0.01,
                            label="Distance Threshold",
                        )

                with gr.Accordion(
                    label="Retrieval Settings",
                    open=False,
                    elem_classes=["accordion"],
                ):
                    top_k = gr.Slider(
                        minimum=1,
                        maximum=20,
                        value=app.top_k,
                        step=1,
                        label="Top K",
                    )

                with gr.Accordion(
                    label="LLM Settings",
                    open=True,
                    elem_classes=["accordion"],
                ):
                    gr.Markdown(CONFIG_MARKDOWN)
                    llm_info = gr.Markdown(configuration_summary())
                    llm_temperature = gr.Slider(
                        minimum=0,
                        maximum=1,
                        value=app.llm_temperature,
                        step=0.1,
                        label="LLM Temperature",
                    )

            with gr.Column(scale=6):
                show_pdf = PDF(
                    label="Uploaded PDF", height=600, elem_classes="pdf-parent"
                )

                with gr.Accordion(
                    label="Chunking Settings",
                    open=False,
                    elem_classes=["accordion"],
                ):
                    chunking_technique = gr.Radio(
                        ["Recursive Character", "Semantic"],
                        label="Chunking Technique",
                        value=app.chunking_technique,
                    )

                    chunk_size = gr.Slider(
                        minimum=100,
                        maximum=2500,
                        value=app.chunk_size,
                        step=50,
                        label="Chunk Size",
                        info="Size of document chunks for processing",
                    )

                with gr.Row():
                    select_pdf_btn = gr.Button("Select PDF", elem_id="select-pdf-btn")
                    reset_btn = gr.Button("Reset", elem_id="reset-btn")

        with Modal(visible=False) as history_modal:
            history_display = gr.Markdown("No chat history available.")

        with Modal(visible=False) as pdf_selector_modal:
            gr.Markdown("## Select a PDF")

            with gr.Row():
                pdf_list = gr.Dataframe(
                    headers=["Filename", "Size (KB)", "Upload Date"],
                    datatype=["str", "number", "str"],
                    col_count=(3, "fixed"),
                    interactive=False,
                    wrap=True,
                    show_label=False,
                )

            with gr.Row():
                upload_btn = gr.UploadButton(
                    "Upload New PDF", file_types=[".pdf"], elem_id="upload-pdf-btn"
                )

        txt.submit(
            fn=add_text,
            inputs=[chatbot, txt],
            outputs=[chatbot],
            queue=False,
        ).success(
            fn=get_response,
            inputs=[
                chatbot,
                txt,
                hidden_file,
                distance_threshold,
                top_k,
                llm_temperature,
                session_state,
            ],
            outputs=[chatbot, txt, feedback_markdown, session_state],
        )

        submit_btn.click(
            fn=add_text,
            inputs=[chatbot, txt],
            outputs=[chatbot],
            queue=False,
        ).success(
            fn=get_response,
            inputs=[
                chatbot,
                txt,
                hidden_file,
                distance_threshold,
                top_k,
                llm_temperature,
                session_state,
            ],
            outputs=[chatbot, txt, feedback_markdown, session_state],
        )

        select_pdf_btn.click(fn=show_pdf_selector, outputs=[pdf_selector_modal, pdf_list])

        pdf_list.select(
            fn=handle_pdf_selection,
            inputs=[pdf_list],
            outputs=[show_pdf, chatbot, feedback_markdown, pdf_selector_modal],
        )

        upload_btn.upload(
            fn=render_first,
            inputs=[upload_btn, chunk_size, chunking_technique, session_state],
            outputs=[show_pdf, chatbot, session_state],
        ).success(
            fn=lambda: (gr.update(visible=False), update_pdf_list()),
            outputs=[pdf_selector_modal, pdf_list],
        )

        reset_btn.click(
            fn=reset_app,
            inputs=None,
            outputs=[
                chatbot,
                show_pdf,
                txt,
                feedback_markdown,
                chunk_size,
                chunking_technique,
                use_semantic_cache,
                top_k,
                distance_threshold,
                llm_temperature,
                use_chat_history,
                use_semantic_router,
                use_ragas,
            ],
        )

        use_chat_history.change(
            fn=app.update_chat_history,
            inputs=[chatbot, use_chat_history, session_state],
            outputs=[chatbot, session_state],
        )

        show_history_btn.click(
            fn=show_history,
            inputs=[session_state],
            outputs=[history_display, history_modal],
        )

        use_semantic_router.change(
            fn=app.update_semantic_router,
            inputs=[use_semantic_router],
            outputs=[],
        )

        use_ragas.change(
            fn=app.update_ragas,
            inputs=[use_ragas],
            outputs=[],
        )

        use_semantic_cache.change(
            fn=app.update_semantic_cache, inputs=[use_semantic_cache], outputs=[]
        )

        blocks.load(lambda: [startup_notice(), configuration_summary()], outputs=[feedback_markdown, llm_info])

    return blocks
