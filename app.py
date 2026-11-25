"""GDPR Assistant: offline RAG chatbot with an ontology (Streamlit + Ollama + ChromaDB).

Run:  streamlit run app.py
"""
import html
import time

import streamlit as st

import rag_core as core

DEFAULT_TOP_K = 5

EXAMPLES = [
    ("⏱️", "Breach notification", "What is the deadline to notify a personal data breach?"),
    ("📋", "All data subject rights", "List all the rights that a data subject has under the GDPR."),
    ("✅", "Valid consent", "What are the conditions for valid consent?"),
    ("💶", "Fines", "What are the maximum administrative fines?"),
]

st.set_page_config(page_title="GDPR Assistant", page_icon="⚖️", layout="centered")

st.markdown(
    """
<style>
#MainMenu, footer {visibility: hidden;}
.block-container {padding-top: 2rem; max-width: 820px;}
.hero {text-align: center; padding: 1.2rem 0 0.4rem 0;}
.hero h1 {font-size: 2.3rem; margin-bottom: 0.2rem;}
.hero p {color: #6b7280; font-size: 1.02rem; margin-top: 0;}
.badge {display: inline-block; background: #e8f1fb; color: #1f4e79; border-radius: 999px;
        padding: 2px 12px; font-size: 0.78rem; font-weight: 600; margin: 0 3px 8px 3px;}
.chip {display: inline-block; background: #eef2ff; color: #3730a3; border: 1px solid #c7d2fe;
       border-radius: 8px; padding: 2px 9px; font-size: 0.8rem; margin: 2px 4px 2px 0;}
div[data-testid="stButton"] > button {border-radius: 12px; text-align: left; height: 3.2rem;}
</style>
""",
    unsafe_allow_html=True,
)


@st.cache_resource
def get_collection():
    return core.get_collection()


@st.cache_resource
def get_ontology():
    try:
        return core.load_ontology()
    except Exception:
        return None


def index_size():
    try:
        return get_collection().count()
    except Exception:
        return None


def show_extras(docs, metas, facts, info):
    """Source chips, ontology facts, response time and retrieved chunks under an answer."""
    labels = list(dict.fromkeys(m["label"] for m in (metas or [])))
    if labels:
        chips = " ".join(f'<span class="chip">{html.escape(l)}</span>' for l in labels)
        st.markdown(f"**Sources:** {chips}", unsafe_allow_html=True)
    if info:
        st.caption(info)
    if facts:
        with st.expander(f"Ontology facts used ({len(facts)})"):
            for f in facts:
                st.markdown(f"- {f.cited()}")
    if docs:
        with st.expander("View retrieved text"):
            for d, m in zip(docs, metas):
                st.markdown(f"**{m['label']}** - {m['title']}")
                st.write(d[:500] + ("..." if len(d) > 500 else ""))


def set_pending(q: str):
    st.session_state.pending = q


if "messages" not in st.session_state:
    st.session_state.messages = []

onto = get_ontology()

# ------------------------------------------------------------------ sidebar
with st.sidebar:
    st.markdown("## ⚖️ GDPR Assistant")
    st.caption("Offline RAG + ontology")
    n = index_size()
    if n:
        st.success(f"Index ready: {n} chunks")
    else:
        st.error("No index found. Run `python ingest.py`.")
    if onto:
        st.success(f"Ontology loaded: {onto.summary()}")
    else:
        st.warning("No ontology found (ontology/gdpr.owl). Only Plain RAG is available.")

    options = list(core.CONFIGS) if onto else ["plain"]
    mode = st.radio("Mode", options, format_func=lambda k: core.CONFIGS[k],
                    help="Plain: vector search + LLM.  Hybrid: the ontology adds facts and articles.  "
                         "Ontology only: no LLM, answers come straight from the ontology.")
    top_k = st.slider("Passages to retrieve", 3, 8, DEFAULT_TOP_K,
                      help="How many GDPR passages are given to the model.", disabled=(mode == "ontology"))
    st.markdown("---")
    st.markdown("**How it works**")
    st.markdown("1. The question is matched to ontology concepts and to GDPR passages\n"
                "2. The ontology adds structured facts (rights, obligations, deadlines, fines)\n"
                "3. A local LLM answers with citations, or the ontology answers directly")
    st.markdown("---")
    st.caption(f"LLM: `{core.LLM_MODEL}`  \nEmbeddings: `{core.EMBED_MODEL}`  \nRuns fully on your computer.")
    if st.button("🧹 Clear chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()
    st.markdown("---")
    st.caption("⚠️ Educational project. Not legal advice.")

# ------------------------------------------------------------------ welcome screen
if not st.session_state.messages:
    st.markdown(
        """
<div class="hero">
  <h1>⚖️ GDPR Assistant</h1>
  <p>Ask questions about the EU General Data Protection Regulation and get answers with article citations.</p>
  <span class="badge">100% offline</span><span class="badge">Cited answers</span><span class="badge">Ontology-powered</span>
</div>
""",
        unsafe_allow_html=True,
    )
    st.markdown("##### Try an example")
    cols = st.columns(2)
    for i, (icon, title, q) in enumerate(EXAMPLES):
        with cols[i % 2]:
            st.button(f"{icon}  {title}", key=f"ex{i}", use_container_width=True,
                      on_click=set_pending, args=(q,), help=q)

# ------------------------------------------------------------------ chat history
for m in st.session_state.messages:
    with st.chat_message(m["role"], avatar="🧑‍💻" if m["role"] == "user" else "⚖️"):
        st.markdown(m["content"])
        if m["role"] == "assistant":
            show_extras(m.get("docs"), m.get("metas"), m.get("facts"), m.get("info"))

# ------------------------------------------------------------------ new question
question = st.chat_input("Ask about the GDPR... (e.g. What is the right to erasure?)")
if st.session_state.get("pending"):
    question = st.session_state.pop("pending")

if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user", avatar="🧑‍💻"):
        st.markdown(question)

    t0 = time.time()
    facts, docs, metas, answer = [], [], [], ""
    try:
        with st.spinner("Searching..."):
            if mode == "ontology":
                answer, docs, metas, facts = core.ontology_only_answer(get_collection(), onto, question)
            elif mode == "hybrid":
                docs, metas, facts = core.retrieve_hybrid(get_collection(), onto, question, top_k)
            else:
                docs, metas = core.retrieve_plain(get_collection(), question, top_k)
    except Exception as e:
        st.error(f"Index not found or Ollama is not running. Run `python ingest.py` first. ({e})")
        st.stop()

    with st.chat_message("assistant", avatar="⚖️"):
        box = st.empty()
        if mode == "ontology":
            box.markdown(answer)
        else:
            msgs = core.build_messages(question, docs, metas, facts if mode == "hybrid" else None)
            for part in core.chat(msgs, stream=True):
                answer += part["message"]["content"]
                box.markdown(answer + "▌")
            box.markdown(answer)
        info = f"{core.CONFIGS[mode]} · answered in {time.time() - t0:.1f} s · {len(docs)} passages, {len(facts)} ontology facts"
        show_extras(docs, metas, facts, info)

    st.session_state.messages.append(
        {"role": "assistant", "content": answer, "docs": docs, "metas": metas, "facts": facts, "info": info})
