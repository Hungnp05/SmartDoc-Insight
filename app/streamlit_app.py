"""
SmartDoc-Insight — Streamlit Application
Layer D: Presentation Layer
────────────────────────────
Features:
  - PDF/Image upload with real-time processing progress
  - Chat interface with streaming responses
  - Source citations with content type indicators
  - Document management sidebar
  - System status monitoring
"""

import sys
import html
import logging
from pathlib import Path

import streamlit as st


sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import config
from src.pipeline import SmartDocPipeline


st.set_page_config(
    page_title="SmartDoc-Insight",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Inter:wght@300;400;500;600&display=swap');

:root {
    --bg-primary: #0d0e14;
    --bg-secondary: #13141e;
    --bg-card: #1a1b27;
    --accent-cyan: #00e5ff;
    --accent-purple: #7c3aed;
    --accent-green: #10b981;
    --accent-amber: #f59e0b;
    --text-primary: #e8eaf0;
    --text-secondary: #8b8fa8;
    --border: #2a2d3e;
}

.stApp {
    background: var(--bg-primary);
    font-family: 'Inter', sans-serif;
    color: var(--text-primary);
}

/* Header */
.smartdoc-header {
    background: linear-gradient(135deg, #0d0e14 0%, #1a1b27 50%, #0d1117 100%);
    border-bottom: 1px solid var(--border);
    padding: 1.5rem 2rem;
    margin: -1rem -1rem 2rem -1rem;
    display: flex;
    align-items: center;
    gap: 1rem;
}

.smartdoc-logo {
    font-family: 'Space Mono', monospace;
    font-size: 1.5rem;
    font-weight: 700;
    background: linear-gradient(90deg, var(--accent-cyan), var(--accent-purple));
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    letter-spacing: -0.02em;
}

.smartdoc-tagline {
    font-size: 0.75rem;
    color: var(--text-secondary);
    font-family: 'Space Mono', monospace;
    letter-spacing: 0.05em;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background: var(--bg-secondary) !important;
    border-right: 1px solid var(--border);
}

/* Chat messages */
.chat-message-user {
    background: linear-gradient(135deg, #1e1f2e, #252640);
    border: 1px solid var(--border);
    border-radius: 12px 12px 4px 12px;
    padding: 1rem 1.25rem;
    margin: 0.5rem 0;
    border-left: 3px solid var(--accent-cyan);
}

.chat-message-assistant {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 4px 12px 12px 12px;
    padding: 1rem 1.25rem;
    margin: 0.5rem 0;
    border-left: 3px solid var(--accent-purple);
    line-height: 1.7;
}

/* Source chips */
.source-chip {
    display: inline-flex;
    align-items: center;
    gap: 0.4rem;
    background: var(--bg-secondary);
    border: 1px solid var(--border);
    border-radius: 6px;
    padding: 0.35rem 0.75rem;
    font-size: 0.75rem;
    font-family: 'Space Mono', monospace;
    color: var(--text-secondary);
    margin: 0.25rem;
    cursor: pointer;
    transition: all 0.2s;
}

.source-chip:hover {
    border-color: var(--accent-cyan);
    color: var(--accent-cyan);
}

.source-chip.table { border-color: var(--accent-amber); color: var(--accent-amber); }
.source-chip.figure { border-color: var(--accent-green); color: var(--accent-green); }

/* Status badge */
.status-badge {
    display: inline-flex;
    align-items: center;
    gap: 0.4rem;
    padding: 0.25rem 0.75rem;
    border-radius: 20px;
    font-size: 0.7rem;
    font-family: 'Space Mono', monospace;
    font-weight: 700;
    letter-spacing: 0.05em;
}

.status-online { background: rgba(16, 185, 129, 0.15); color: var(--accent-green); border: 1px solid rgba(16, 185, 129, 0.3); }
.status-offline { background: rgba(239, 68, 68, 0.15); color: #ef4444; border: 1px solid rgba(239, 68, 68, 0.3); }

/* Processing steps */
.process-step {
    display: flex;
    align-items: center;
    gap: 0.75rem;
    padding: 0.6rem 1rem;
    border-radius: 8px;
    margin: 0.4rem 0;
    font-size: 0.85rem;
    border: 1px solid var(--border);
}

.step-active { background: rgba(0, 229, 255, 0.05); border-color: var(--accent-cyan); }
.step-done { background: rgba(16, 185, 129, 0.05); border-color: var(--accent-green); }

/* Metric cards */
.metric-card {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 1rem;
    text-align: center;
}

.metric-value {
    font-family: 'Space Mono', monospace;
    font-size: 1.5rem;
    font-weight: 700;
    color: var(--accent-cyan);
}

.metric-label {
    font-size: 0.7rem;
    color: var(--text-secondary);
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin-top: 0.25rem;
}

/* Buttons */
.stButton > button {
    background: linear-gradient(135deg, var(--accent-cyan), var(--accent-purple)) !important;
    color: #000 !important;
    font-family: 'Space Mono', monospace !important;
    font-weight: 700 !important;
    border: none !important;
    border-radius: 8px !important;
    letter-spacing: 0.03em;
    transition: all 0.2s !important;
}

.stButton > button:hover {
    opacity: 0.9 !important;
    transform: translateY(-1px) !important;
    box-shadow: 0 4px 20px rgba(0, 229, 255, 0.3) !important;
}

/* File uploader */
.stFileUploader {
    border: 2px dashed var(--border) !important;
    border-radius: 12px !important;
    background: var(--bg-card) !important;
    transition: border-color 0.2s !important;
}

.stFileUploader:hover {
    border-color: var(--accent-cyan) !important;
}

/* Input */
.stTextInput > div > div > input, .stChatInput {
    background: var(--bg-card) !important;
    border: 1px solid var(--border) !important;
    color: var(--text-primary) !important;
    border-radius: 10px !important;
    font-family: 'Inter', sans-serif !important;
}

/* Markdown tables */
table { width: 100%; border-collapse: collapse; font-size: 0.85rem; }
th { background: var(--bg-secondary); color: var(--accent-cyan); padding: 0.5rem 0.75rem; border: 1px solid var(--border); font-family: 'Space Mono', monospace; font-size: 0.75rem; }
td { padding: 0.5rem 0.75rem; border: 1px solid var(--border); color: var(--text-primary); }
tr:hover td { background: rgba(255,255,255,0.02); }

/* Expanders */
.streamlit-expanderHeader {
    background: var(--bg-card) !important;
    border: 1px solid var(--border) !important;
    border-radius: 8px !important;
    font-family: 'Space Mono', monospace !important;
    font-size: 0.8rem !important;
}

/* Divider */
hr { border-color: var(--border) !important; }

/* Scrollbar */
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: var(--bg-primary); }
::-webkit-scrollbar-thumb { background: var(--border); border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: var(--accent-cyan); }
</style>
""", unsafe_allow_html=True)


# Session State Init

def init_session():
    defaults = {
        "messages": [],
        "active_doc": None,
        "confirm_clear": False,
        "top_k": config.chroma.top_k,
        "rerank_k": config.chroma.rerank_top_k,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


# System Initialization

@st.cache_resource(show_spinner=False)
def load_pipeline() -> SmartDocPipeline:
    """Load all RAG components once per server process (shared by every session)."""
    return SmartDocPipeline(config)


# Sidebar

def render_sidebar(pipeline: SmartDocPipeline):
    with st.sidebar:
        st.markdown("### SmartDoc-Insight")
        st.markdown("---")

        # System Status
        st.markdown("#### System Status")
        status = pipeline.check_models()

        for name, ok, model in [
            ("LLM", status["llm"], config.ollama.llm_model),
            ("Vision", status["vision"], config.ollama.vision_model),
            ("Embed", status["embed"], config.ollama.embed_model),
        ]:
            badge_class = "status-online" if ok else "status-offline"
            icon = "●" if ok else "○"
            st.markdown(
                f'<div class="status-badge {badge_class}">{icon} {html.escape(name)}: {html.escape(model)}</div>',
                unsafe_allow_html=True
            )

        if not all([status["llm"], status["vision"], status["embed"]]):
            missing = [
                m for m, ok in [
                    (config.ollama.llm_model, status["llm"]),
                    (config.ollama.vision_model, status["vision"]),
                    (config.ollama.embed_model, status["embed"]),
                ] if not ok
            ]
            commands = "\n".join(f"ollama pull {m}" for m in missing)
            st.warning(f"⚠️ Ollama is not running or models are missing. Run:\n```\n{commands}\n```")

        st.markdown("---")

        # Upload
        st.markdown("#### Upload Document")
        uploaded_file = st.file_uploader(
            "PDF or Image",
            type=config.app.allowed_extensions,
            label_visibility="collapsed",
        )

        if uploaded_file:
            if pipeline.kb.has_document(uploaded_file.name):
                st.info(f"`{uploaded_file.name}` is already indexed; processing replaces it.")
            if st.button("Process Document", use_container_width=True):
                process_document(uploaded_file, pipeline)

        st.markdown("---")

        # Knowledge Base Stats
        st.markdown("#### Knowledge Base")
        stats = pipeline.stats()

        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{stats['total_chunks']}</div>
                <div class="metric-label">Chunks</div>
            </div>""", unsafe_allow_html=True)
        with col2:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{stats['total_documents']}</div>
                <div class="metric-label">Docs</div>
            </div>""", unsafe_allow_html=True)

        if stats['documents']:
            st.markdown("**Indexed Documents:**")
            for doc in stats['documents']:
                with st.expander(f"📄 {doc['source_file']}", expanded=False):
                    st.markdown(f"- Pages: {doc['total_pages']}")
                    st.markdown(f"- Chunks: {doc['chunk_count']}")

                    # Filter option
                    if st.button("Chat with this doc", key=f"filter_{doc['source_file']}"):
                        st.session_state.active_doc = doc['source_file']
                        st.rerun()

        if st.session_state.active_doc:
            st.info(f"Filtering: `{st.session_state.active_doc}`")
            if st.button("❌ Clear Filter"):
                st.session_state.active_doc = None
                st.rerun()

        st.markdown("---")

        # Settings (per session; the shared config is not modified)
        st.markdown("#### ⚙️ Settings")
        st.slider("Retrieve Top-K", 3, 15, key="top_k")
        st.slider("Re-rank Top-K", 2, 8, key="rerank_k")

        if not st.session_state.confirm_clear:
            if st.button("🗑️ Clear Knowledge Base", use_container_width=True):
                st.session_state.confirm_clear = True
                st.rerun()
        else:
            st.warning("Delete every indexed document?")
            col_yes, col_no = st.columns(2)
            if col_yes.button("⚠️ Confirm", key="confirm_clear_yes", use_container_width=True):
                pipeline.clear()
                st.session_state.confirm_clear = False
                st.session_state.active_doc = None
                st.toast("Knowledge base cleared!")
                st.rerun()
            if col_no.button("Cancel", key="confirm_clear_no", use_container_width=True):
                st.session_state.confirm_clear = False
                st.rerun()

        st.markdown("---")
        st.markdown(
            '<div style="text-align:center;font-size:0.65rem;color:#4a4d66;font-family:Space Mono">'
            'SmartDoc-Insight v1.0<br>Local-First · Privacy-Preserving</div>',
            unsafe_allow_html=True
        )


# Document Processing

def process_document(uploaded_file, pipeline: SmartDocPipeline):
    """Handle document upload and processing pipeline."""
    save_path = config.paths["uploads"] / Path(uploaded_file.name).name
    with open(save_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    with st.sidebar:
        st.markdown("**Processing Pipeline:**")
        status_text = st.empty()
        progress_bar = st.progress(0.0)

    def on_progress(msg, fraction):
        status_text.markdown(msg)
        if fraction is not None:
            progress_bar.progress(min(max(fraction, 0.0), 1.0))

    try:
        result = pipeline.ingest(save_path, progress_callback=on_progress)
    except Exception as e:
        st.sidebar.error(f"❌ Processing failed: {e}")
        logging.exception("Document processing error")
        return

    if result["total_chunks"] == 0:
        st.sidebar.warning(
            f"⚠️ No content could be extracted from '{uploaded_file.name}'. "
            "If it is a scan or an image, install OCR support (requirements-ocr.txt)."
        )
        return

    status_text.markdown(f"✅ Done! {result['total_chunks']} chunks indexed")
    type_summary = " | ".join(f"{k}: {v}" for k, v in result["regions"].items())
    st.sidebar.success(f"✅ '{uploaded_file.name}' processed: {result['total_chunks']} chunks")
    st.sidebar.info(f"Regions detected: {type_summary}")


# Chat Interface

def render_chat(pipeline: SmartDocPipeline):
    """Main chat area with message history and streaming responses."""

    # Header
    st.markdown("""
    <div class="smartdoc-header"
        style="display: flex; justify-content: center; width: 100%;">
        <div style="text-align: center;">
            <div class="smartdoc-logo">SmartDoc-Insight</div>
            <div class="smartdoc-tagline">MULTI-MODAL RAG · LOCAL-FIRST · TABLE-AWARE</div>
            <div style="font-family: Space Mono; font-size: 0.9rem; color: #6b6f88; margin-bottom: 0.5rem;">
                Upload a PDF to begin. I can analyze text, tables, and charts.
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Message History (rendered as Markdown, never as raw HTML)
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("sources"):
                render_sources(msg["sources"])

    prompt = st.chat_input("Ask questions about the document....")
    if not prompt:
        return

    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        if pipeline.stats()["total_chunks"] == 0:
            answer = "Please upload the document before asking a question."
            st.markdown(answer)
            st.session_state.messages.append({"role": "assistant", "content": answer, "sources": []})
            return

        try:
            with st.spinner("Searching the document..."):
                response = pipeline.ask_stream(
                    prompt,
                    source_filter=st.session_state.active_doc,
                    top_k=st.session_state.top_k,
                    rerank_top_k=st.session_state.rerank_k,
                )
            st.caption(
                f"Retrieved: {response.retrieved_count} → "
                f"Re-ranked: {response.reranked_count} chunks"
            )
            answer = st.write_stream(response.tokens)
            render_sources(response.sources)
            st.session_state.messages.append({
                "role": "assistant",
                "content": answer,
                "sources": response.sources,
            })
        except Exception as e:
            error_msg = f"⚠️ Error generating response: {e}"
            st.error(error_msg)
            st.session_state.messages.append({"role": "assistant", "content": error_msg, "sources": []})


def render_sources(sources):
    """Display source citations below an answer."""
    if not sources:
        return

    st.markdown("**📎 References:**")

    chips = []
    for source in sources:
        type_class = source.content_type if source.content_type in ("table", "figure") else "text"
        chips.append(
            f'<span class="source-chip {type_class}">'
            f'Trang {source.page} · {html.escape(source.content_type.upper())} '
            f'· {source.score:.0%}'
            f'</span>'
        )
    st.markdown("".join(chips), unsafe_allow_html=True)

    # Expandable source details
    with st.expander("See the detailed source content", expanded=False):
        for i, source in enumerate(sources, 1):
            st.markdown(f"**{i}. {source.display_label}** — `{source.source_file}` (Score: {source.score:.4f})")
            st.markdown(source.content)
            st.markdown("---")


# Main

def main():
    init_session()

    try:
        pipeline = load_pipeline()
    except Exception as e:
        st.error(f"Failed to initialize system: {e}")
        st.info("Make sure Ollama is running: `ollama serve`")
        return

    render_sidebar(pipeline)

    # Main content tabs
    tab_chat, tab_debug, tab_about = st.tabs(["💬 Chat", "🔬 Debug", "ℹ️ About"])

    with tab_chat:
        render_chat(pipeline)

    with tab_debug:
        render_debug(pipeline)

    with tab_about:
        render_about()


def render_debug(pipeline: SmartDocPipeline):
    """Debug view: show retrieval pipeline internals."""
    st.markdown("### Retrieval Debug")
    st.markdown("Test the retrieval pipeline and see re-ranking in action.")
    if st.session_state.active_doc:
        st.caption(f"Filtering: `{st.session_state.active_doc}`")

    debug_query = st.text_input("Test Query:", placeholder="Enter a test question...")
    if st.button("Run Debug Retrieval") and debug_query:
        with st.spinner("Running retrieval..."):
            try:
                debug_info = pipeline.rag.get_retrieval_debug(
                    debug_query,
                    filter_source=st.session_state.active_doc,
                    top_k=st.session_state.top_k,
                    rerank_top_k=st.session_state.rerank_k,
                )
            except Exception as e:
                st.error(f"Debug failed: {e}")
                return

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Initial Retrieval (by vector similarity):**")
            for item in debug_info["initial_retrieval"]:
                st.markdown(f"""
                <div style="background:#13141e;border:1px solid #2a2d3e;border-radius:8px;padding:0.75rem;margin:0.4rem 0">
                    <div style="font-family:Space Mono;font-size:0.65rem;color:#00e5ff">
                        TYPE: {html.escape(str(item['type']))} | PAGE: {item['page']} | SCORE: {item['score']:.4f}
                    </div>
                    <div style="font-size:0.8rem;color:#8b8fa8;margin-top:0.3rem">{html.escape(item['text'])}...</div>
                </div>
                """, unsafe_allow_html=True)

        with col2:
            st.markdown("**After Re-ranking (cross-encoder):**")
            for item in debug_info["after_reranking"]:
                improvement = item['rerank_score'] - item['original_score']
                color = "#10b981" if improvement > 0 else "#ef4444"
                arrow = "↑" if improvement > 0 else "↓"
                st.markdown(f"""
                <div style="background:#13141e;border:1px solid #2a2d3e;border-radius:8px;padding:0.75rem;margin:0.4rem 0">
                    <div style="font-family:Space Mono;font-size:0.65rem;color:#7c3aed">
                        RERANK: {item['rerank_score']:.4f}
                        <span style="color:{color}"> {arrow}{abs(improvement):.4f}</span>
                    </div>
                    <div style="font-size:0.8rem;color:#8b8fa8;margin-top:0.3rem">{html.escape(item['text'])}...</div>
                </div>
                """, unsafe_allow_html=True)


def render_about():
    """About page with architecture overview."""
    st.markdown("""
    ### SmartDoc-Insight

    **Multi-Modal RAG for Complex Documents** — Local-First, Privacy-Preserving.

    #### Architecture

    | Layer | Component | Purpose |
    |-------|-----------|---------|
    | A - Vision | PyMuPDF + LLaVA-7B (+ optional PaddleOCR) | Table detection → Markdown, chart → text, OCR for scans/images |
    | B - Knowledge | ChromaDB + nomic-embed-text | Vector store with enriched metadata |
    | C - Reasoning | Cross-encoder + Llama3-8B | Re-rank + generate answers |
    | D - Interface | Streamlit | This UI |

    #### Tech Stack
    `Python` · `PyMuPDF` · `LLaVA-7B` · `Llama3-8B` · `ChromaDB` · `sentence-transformers` · `Ollama` · `Streamlit` · `Docker`
    """)


if __name__ == "__main__":
    main()
