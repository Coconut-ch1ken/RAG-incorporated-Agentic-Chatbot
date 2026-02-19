"""
Streamlit Web UI for the Personal Assistant RAG Chatbot.
Run with: .venv/bin/streamlit run streamlit_app.py
"""
from dotenv import load_dotenv
load_dotenv()  # Load .env before anything else needs GOOGLE_API_KEY

import streamlit as st
from src.graph.workflow import RagAgent
from src.ingestion.directory_scanner import DirectoryScanner
from src.database.vector_store import VectorStore
from src.database.metadata_store import MetadataStore
from src.config import settings

# ---- Page Config ----
st.set_page_config(
    page_title="RAG Personal Assistant",
    page_icon="🤖",
    layout="wide",
)

# ---- Custom CSS ----
st.markdown("""
<style>
    /* Dark theme overrides */
    .stApp {
        background-color: #0e1117;
    }

    /* Tier badges */
    .tier-badge {
        display: inline-block;
        padding: 2px 10px;
        border-radius: 12px;
        font-size: 0.75em;
        font-weight: 600;
        margin-bottom: 4px;
    }
    .tier-local { background: #1a3a2a; color: #4ade80; }
    .tier-local-gemini { background: #1a2a3a; color: #60a5fa; }
    .tier-gemini { background: #2a1a3a; color: #c084fc; }
    .tier-powerful { background: #3a2a1a; color: #fbbf24; }

    /* Sidebar styling */
    .file-item {
        padding: 4px 8px;
        margin: 2px 0;
        background: #1e2530;
        border-radius: 6px;
        font-size: 0.85em;
    }

    /* Scan stats */
    .scan-stats {
        padding: 8px 12px;
        background: #1a2a1a;
        border-radius: 8px;
        border-left: 3px solid #4ade80;
        margin: 8px 0;
    }
</style>
""", unsafe_allow_html=True)


# ---- Initialize Session State ----
if "messages" not in st.session_state:
    st.session_state.messages = []

if "agent" not in st.session_state:
    with st.spinner("🔄 Initializing RAG Agent..."):
        st.session_state.agent = RagAgent()

if "vector_store" not in st.session_state:
    st.session_state.vector_store = st.session_state.agent.vector_store

if "metadata_store" not in st.session_state:
    st.session_state.metadata_store = MetadataStore()


# ---- Sidebar ----
with st.sidebar:
    st.title("📂 知识库管理")
    st.divider()

    # Scan button
    if st.button("🔍 扫描数据目录", use_container_width=True):
        with st.spinner(f"扫描 `{settings.watch_directory}` 中..."):
            scanner = DirectoryScanner(
                st.session_state.vector_store,
                st.session_state.metadata_store,
            )
            stats = scanner.scan(settings.watch_directory, user_id="default_user")

        st.markdown(
            f'<div class="scan-stats">'
            f'✅ 摄入: <b>{stats["ingested"]}</b> &nbsp;|&nbsp; '
            f'⏩ 跳过: <b>{stats["skipped"]}</b> &nbsp;|&nbsp; '
            f'❌ 错误: <b>{stats["errors"]}</b>'
            f'</div>',
            unsafe_allow_html=True,
        )

    st.divider()

    # File list
    st.subheader("📄 已摄入文件")
    files = st.session_state.metadata_store.get_user_files("default_user")
    if files:
        unique_files = sorted(set(files))
        for f in unique_files:
            st.markdown(f'<div class="file-item">📄 {f}</div>', unsafe_allow_html=True)
    else:
        st.caption("还没有摄入任何文件，请先点击上方扫描按钮。")

    st.divider()

    # Clear chat
    if st.button("🗑️ 清空聊天", use_container_width=True):
        st.session_state.messages = []
        st.rerun()


# ---- Main Chat Area ----
st.title("🤖 RAG Personal Assistant")
st.caption("基于 LangGraph + Ollama + Gemini 的个人知识助手")

# Display chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        if msg["role"] == "assistant" and "tier" in msg:
            tier = msg["tier"]
            tier_map = {
                "local": ("🏠 Local", "tier-local"),
                "local+gemini": ("🏠+☁️ Local+Gemini", "tier-local-gemini"),
                "powerful": ("💪 Powerful", "tier-powerful"),
                "gemini": ("☁️ Gemini", "tier-gemini"),
            }
            label, css_class = tier_map.get(tier, (f"❓ {tier}", "tier-local"))
            st.markdown(
                f'<span class="tier-badge {css_class}">{label}</span>',
                unsafe_allow_html=True,
            )
        st.markdown(msg["content"])

# Chat input
if prompt := st.chat_input("问点什么？"):
    # Show user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Get AI response
    with st.chat_message("assistant"):
        with st.spinner("思考中..."):
            result = st.session_state.agent.run(prompt, user_id="default_user")

        generation = result.get("generation", "抱歉，我无法生成回答。")
        tier = result.get("generation_tier", "unknown")

        tier_map = {
            "local": ("🏠 Local", "tier-local"),
            "local+gemini": ("🏠+☁️ Local+Gemini", "tier-local-gemini"),
            "powerful": ("💪 Powerful", "tier-powerful"),
            "gemini": ("☁️ Gemini", "tier-gemini"),
        }
        label, css_class = tier_map.get(tier, (f"❓ {tier}", "tier-local"))
        st.markdown(
            f'<span class="tier-badge {css_class}">{label}</span>',
            unsafe_allow_html=True,
        )
        st.markdown(generation)

    st.session_state.messages.append({
        "role": "assistant",
        "content": generation,
        "tier": tier,
    })
