import os
import time
from dotenv import load_dotenv
import pandas as pd
import plotly.express as px
import streamlit as st

from src.agent.financial_agent import FinancialAgent
from src.ingestion import ingest_pdf
from src.storage import chat_history, document_manifest

load_dotenv()
chat_history.init_db()


st.set_page_config(
    page_title="LucaAi",
    page_icon="📈",
    layout="wide"
)

st.title("📈 LucaAi")
st.caption(
    "AI agent for financial analysis based on RAG and real-time market data")

WELCOME_MESSAGE = "Hi! What financial information or stock analysis can I help you with today?"

if "eval_history" not in st.session_state:
    st.session_state.eval_history = []


def log_evaluation_metric(query_label: str, latency: float, iterations: int, evaluation: dict) -> None:
    st.session_state.eval_history.append({
        "Query": query_label,
        "Latency (s)": round(latency, 2),
        "Iterations": iterations,
        "Evidence coverage (%)": evaluation.get("score"),
        "Claims checked": evaluation.get("checked_claims", 0),
        "Supported claims": evaluation.get("supported_claims", 0),
    })


def start_new_chat() -> None:
    st.session_state.conversation_id = None
    st.session_state.messages = [
        {"role": "assistant", "content": WELCOME_MESSAGE}]
    st.session_state.agent = FinancialAgent()


def load_chat(conversation_id: str) -> None:
    st.session_state.conversation_id = conversation_id
    st.session_state.messages = chat_history.get_messages(conversation_id)
    st.session_state.agent = FinancialAgent()


if "messages" not in st.session_state:
    start_new_chat()

with st.sidebar:
    st.header("💬 Conversations")

    if st.button("+ New chat", use_container_width=True):
        start_new_chat()
        st.rerun()

    conversations = chat_history.list_conversations()
    if conversations:
        for conv in conversations:
            col_title, col_delete = st.columns([5, 1])
            is_active = conv["id"] == st.session_state.get("conversation_id")
            with col_title:
                if st.button(
                    ("🟢 " if is_active else "") + conv["title"],
                    key=f"load_{conv['id']}",
                    use_container_width=True,
                ):
                    load_chat(conv["id"])
                    st.rerun()
            with col_delete:
                if st.button("✕", key=f"delete_{conv['id']}"):
                    chat_history.delete_conversation(conv["id"])
                    if is_active:
                        start_new_chat()
                    st.rerun()
    else:
        st.caption("No saved conversations yet.")

    st.divider()
    st.header("📂 Data Ingestion")

    uploaded_files = st.file_uploader(
        "Upload PDF reports (e.g. 10-K, 10-Q) — multiple allowed",
        type=["pdf"], accept_multiple_files=True,
    )

    if uploaded_files:
        for uf in uploaded_files:
            save_path = os.path.join("data", "raw_pdfs", uf.name)
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            with open(save_path, "wb") as f:
                f.write(uf.getbuffer())

    pending = ingest_pdf.list_pending_pdfs()
    if pending:
        st.caption(
            f"📋 {len(pending)} file(s) ready to index: {', '.join(pending)}")

        if st.button("🚀 Process and Index", use_container_width=True):
            progress_bar = st.progress(0.0)

            stage_order = ["loading", "chunking", "embedding", "file_done"]
            stage_labels = {"loading": "📄 Load", "chunking": "🧩 Chunk",
                            "embedding": "🧠 Embed", "file_done": "💾 Store"}
            stage_cols = st.columns(4)
            stage_placeholders = [c.empty() for c in stage_cols]
            status_text = st.empty()

            def render_pipeline(active_stage: str) -> None:
                reached = stage_order.index(active_stage)
                for i, stage in enumerate(stage_order):
                    icon = "✅" if i < reached else "🔄" if i == reached else "⚪"
                    stage_placeholders[i].markdown(
                        f"<div style='text-align:center'>{icon}<br><small>{stage_labels[stage]}</small></div>", unsafe_allow_html=True)

            for event in ingest_pdf.process_and_store_pdfs():
                stage = event["stage"]
                if stage == "nothing_to_do":
                    status_text.info("Nothing new to index.")
                elif stage == "error":
                    status_text.error(
                        f"⚠️ Failed on {event['filename']}: {event['error']}")
                elif stage in ("loading", "chunking", "embedding"):
                    render_pipeline(stage)
                    status_text.caption(
                        f"{event['filename']} — file {event['current']}/{event['total']}")
                elif stage == "file_done":
                    render_pipeline("file_done")
                    status_text.caption(
                        f"✅ {event['filename']}: {event['chunk_count']} chunks indexed")
                    progress_bar.progress(event["current"] / event["total"])
                elif stage == "all_done":
                    status_text.success(
                        f"Indexing complete — {event['total']} file(s) processed.")

            st.rerun()
    elif uploaded_files:
        st.caption("All uploaded files are already indexed.")

    indexed_docs = document_manifest.list_ingested()
    if indexed_docs:
        st.caption("📚 Indexed documents:")
        for doc in indexed_docs:
            col_name, col_del = st.columns([4, 1])
            with col_name:
                st.caption(f"{doc['filename']} · {doc['chunk_count']} chunks")
            with col_del:
                if st.button("🗑️", key=f"remove_doc_{doc['filename']}"):
                    ingest_pdf.remove_document(doc["filename"])
                    st.rerun()

    st.divider()
    st.header("📊 Real-Time Evaluation")

    if st.session_state.eval_history:
        df_metrics = pd.DataFrame(st.session_state.eval_history)

        col1, col2 = st.columns(2)
        col1.metric("Avg Latency", f"{df_metrics['Latency (s)'].mean():.1f}s")
        col2.metric("Avg Iterations", f"{df_metrics['Iterations'].mean():.1f}")

        scored = df_metrics.dropna(subset=["Evidence coverage (%)"])
        if not scored.empty:
            st.metric(
                "Evidence coverage",
                f"{scored['Evidence coverage (%)'].mean():.0f}%",
                delta="Numeric claims matched to this turn's tool evidence",
            )
        else:
            st.metric("Evidence coverage", "N/A",
                      delta="No checkable numeric claims with tool evidence")

        fig_latency = px.bar(
            df_metrics,
            x="Query",
            y="Latency (s)",
            color="Iterations",
            title="Execution Latency & ReAct Iterations",
            labels={"Latency (s)": "Latency (sec)",
                    "Iterations": "Iterations"},
            color_continuous_scale="Blues",
            text="Latency (s)"
        )
        fig_latency.update_layout(
            height=260, margin=dict(l=10, r=10, t=35, b=10))
        st.plotly_chart(fig_latency, use_container_width=True)
        st.success("✅ Guardrails: Active | Token Budget: Safe")
    else:
        st.caption("No queries run yet. Evaluation metrics will appear here.")


def render_chat_message(content: str) -> None:
    """Streamlit's st.markdown() interprets '$' as LaTeX math delimiters.
    Financial responses are full of dollar amounts, so unescaped '$' signs
    cause garbled/duplicated rendering. Escaping them keeps dollar amounts as
    plain text."""
    st.markdown(content.replace("$", "\\$"))


for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        render_chat_message(message["content"])

if user_input := st.chat_input("Ask a question (e.g. What was Micron's 2025 revenue and MU's current price?)..."):
    conversation_history = st.session_state.messages.copy()

    if st.session_state.conversation_id is None:
        title = chat_history.make_title(user_input)
        st.session_state.conversation_id = chat_history.create_conversation(
            title)
        for msg in st.session_state.messages:
            chat_history.add_message(
                st.session_state.conversation_id, msg["role"], msg["content"])

    st.session_state.messages.append({"role": "user", "content": user_input})
    chat_history.add_message(
        st.session_state.conversation_id, "user", user_input)
    with st.chat_message("user"):
        render_chat_message(user_input)

    with st.chat_message("assistant"):
        with st.spinner("Agent is analyzing the data..."):
            start_time = time.time()

            response_text = st.session_state.agent.run(
                user_input, history=conversation_history)

            end_time = time.time()
            execution_latency = end_time - start_time

            iterations_used = getattr(
                st.session_state.agent, "last_iterations", 2)

            query_short = (
                user_input[:12] + "...") if len(user_input) > 12 else user_input
            evaluation = getattr(st.session_state.agent, "last_evaluation", {})
            log_evaluation_metric(
                query_label=query_short,
                latency=execution_latency,
                iterations=iterations_used,
                evaluation=evaluation,
            )

            render_chat_message(response_text)
            score = evaluation.get("score")
            if score is None:
                st.caption(
                    "Evidence coverage: N/A — no checkable numeric claims or tool evidence for this answer.")
            else:
                st.caption(
                    f"Evidence coverage: {score}% — {evaluation.get('supported_claims', 0)}/"
                    f"{evaluation.get('checked_claims', 0)} numeric claims matched this turn's tool output."
                )
                unsupported = evaluation.get("unsupported_claims", [])
                if unsupported:
                    with st.expander("Claims not matched to retrieved evidence"):
                        for claim in unsupported:
                            st.write(f"- {claim}")

    st.session_state.messages.append(
        {"role": "assistant", "content": response_text})
    chat_history.add_message(
        st.session_state.conversation_id, "assistant", response_text)

    st.rerun()
