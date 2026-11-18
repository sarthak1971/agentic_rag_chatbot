import os
from typing import TypedDict, List

import docx  # python-docx; imported as a module so it doesn't clash with langchain's Document
import streamlit as st
from dotenv import load_dotenv
from pypdf import PdfReader
from langchain_core.documents import Document
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.graph import StateGraph, END

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------
load_dotenv()

st.set_page_config(
    page_title="Agentic RAG Chatbot",
    page_icon=":brain:",
    layout="centered",
)

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    st.error("GROQ_API_KEY not found. Add it to your .env file and restart the app.")
    st.stop()

SAMPLE_TEXTS = [
    "LangGraph is a library for building stateful, multi-actor applications with LLMs. It extends LangChain with the ability to coordinate multiple chains across multiple steps of computation in a cyclic manner.",
    "RAG (Retrieval-Augmented Generation) is a technique that combines information retrieval with text generation. It retrieves relevant documents and uses them to provide context for generating more accurate responses.",
    "Vector databases store high-dimensional vectors and enable efficient similarity search. They are commonly used in RAG systems to find relevant documents based on semantic similarity.",
    "Agentic systems are AI systems that can take actions, make decisions, and interact with their environment autonomously. They often use planning and reasoning capabilities.",
]

RETRIEVAL_KEYWORDS = ["what", "how", "explain", "describe", "tell me"]


# ---------------------------------------------------------------------------
# Cached resources (created once per server session)
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading Groq model...")
def get_llm(model_name: str, temperature: float) -> ChatGroq:
    return ChatGroq(
        api_key=GROQ_API_KEY,
        model=model_name,
        temperature=temperature,
        max_tokens=2048,
    )


@st.cache_resource(show_spinner="Loading embedding model (first run downloads it)...")
def get_embeddings() -> HuggingFaceEmbeddings:
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")


# ---------------------------------------------------------------------------
# Agentic RAG graph (same logic as the notebook)
# ---------------------------------------------------------------------------
class AgentState(TypedDict):
    question: str
    documents: List[Document]
    answer: str
    needs_retrieval: bool


def build_graph(llm: ChatGroq, retriever):
    def decide_retrieval(state: AgentState) -> AgentState:
        question = state["question"].lower()
        needs_retrieval = any(k in question for k in RETRIEVAL_KEYWORDS)
        return {**state, "needs_retrieval": needs_retrieval}

    def retrieve_documents(state: AgentState) -> AgentState:
        documents = retriever.invoke(state["question"])
        return {**state, "documents": documents}

    def generate_answer(state: AgentState) -> AgentState:
        question = state["question"]
        documents = state.get("documents", [])

        if documents:
            context = "\n\n".join(doc.page_content for doc in documents)
            prompt = (
                "Based on the following context, answer the question:\n\n"
                f"Context:\n{context}\n\n"
                f"Question: {question}\n\n"
                "Answer:"
            )
        else:
            prompt = f"Answer the following question: {question}"

        response = llm.invoke(prompt)
        return {**state, "answer": response.content}

    def should_retrieve(state: AgentState) -> str:
        return "retrieve" if state["needs_retrieval"] else "generate"

    workflow = StateGraph(AgentState)
    workflow.add_node("decide", decide_retrieval)
    workflow.add_node("retrieve", retrieve_documents)
    workflow.add_node("generate", generate_answer)
    workflow.set_entry_point("decide")
    workflow.add_conditional_edges(
        "decide",
        should_retrieve,
        {"retrieve": "retrieve", "generate": "generate"},
    )
    workflow.add_edge("retrieve", "generate")
    workflow.add_edge("generate", END)
    return workflow.compile()


def rebuild_pipeline(model_name: str, temperature: float, top_k: int):
    """(Re)build the vector store, retriever and graph from the current documents."""
    embeddings = get_embeddings()
    llm = get_llm(model_name, temperature)
    vectorstore = FAISS.from_documents(st.session_state.docs, embeddings)
    retriever = vectorstore.as_retriever(search_kwargs={"k": top_k})
    st.session_state.graph = build_graph(llm, retriever)
    st.session_state.pipeline_key = (model_name, temperature, top_k, len(st.session_state.docs))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
FILE_TYPES = ["pdf", "txt", "md", "docx", "csv"]


def load_uploaded_file(uploaded) -> List[Document]:
    """Turn an uploaded file (.pdf, .docx, .txt, .md, .csv) into chunked Documents."""
    name = uploaded.name
    lower = name.lower()

    if lower.endswith(".pdf"):
        reader = PdfReader(uploaded)
        raw_docs = [
            Document(page_content=page.extract_text() or "", metadata={"source": name, "page": i + 1})
            for i, page in enumerate(reader.pages)
        ]
    elif lower.endswith(".docx"):
        document = docx.Document(uploaded)
        text = "\n".join(p.text for p in document.paragraphs)
        raw_docs = [Document(page_content=text, metadata={"source": name})]
    else:  # .txt, .md, .csv are read as plain text
        text = uploaded.read().decode("utf-8", errors="ignore")
        raw_docs = [Document(page_content=text, metadata={"source": name})]

    raw_docs = [d for d in raw_docs if d.page_content.strip()]
    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
    return splitter.split_documents(raw_docs)


def add_files(files) -> List[str]:
    """Add files to the knowledge base, skipping ones already added. Returns a status message per file."""
    messages = []
    for f in files:
        file_id = (f.name, f.size)
        if file_id in st.session_state.loaded_files:
            messages.append(f"{f.name}: already added, skipped")
            continue
        try:
            chunks = load_uploaded_file(f)
        except Exception as e:
            messages.append(f"{f.name}: could not be read ({e})")
            continue
        if not chunks:
            messages.append(f"{f.name}: no readable text found (a scanned PDF needs OCR)")
            continue
        st.session_state.docs.extend(chunks)
        st.session_state.loaded_files.add(file_id)
        messages.append(f"{f.name}: added {len(chunks)} chunks")
    return messages


def render_meta(meta: dict):
    if meta["needs_retrieval"]:
        st.caption(f"Route: retrieval used ({len(meta['documents'])} chunks)")
        with st.expander("Retrieved context"):
            for i, doc in enumerate(meta["documents"], 1):
                source = doc.metadata.get("source", "sample knowledge base")
                st.markdown(f"**{i}. {source}**")
                st.write(doc.page_content)
    else:
        st.caption("Route: answered directly (no retrieval)")


# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------
if "docs" not in st.session_state:
    st.session_state.docs = [Document(page_content=t, metadata={"source": "sample knowledge base"}) for t in SAMPLE_TEXTS]
if "messages" not in st.session_state:
    st.session_state.messages = []
if "loaded_files" not in st.session_state:
    st.session_state.loaded_files = set()

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("Settings")
    model_name = st.selectbox(
        "Groq model",
        ["openai/gpt-oss-120b", "llama-3.3-70b-versatile"],
    )
    temperature = st.slider("Temperature", 0.0, 1.0, 0.0, 0.1)
    top_k = st.slider("Chunks to retrieve (k)", 1, 8, 3)

    st.header("Knowledge base")
    st.caption(f"{len(st.session_state.docs)} chunks loaded")
    st.caption("You can also attach files directly in the chat box.")
    uploads = st.file_uploader(
        "Add files (PDF, DOCX, TXT, MD, CSV)", type=FILE_TYPES, accept_multiple_files=True
    )
    if st.button("Add to knowledge base", disabled=not uploads):
        with st.spinner("Reading and splitting files..."):
            for line in add_files(uploads):
                st.write(line)
        st.session_state.pipeline_key = None  # force rebuild

    if st.button("Reset knowledge base"):
        st.session_state.docs = [Document(page_content=t, metadata={"source": "sample knowledge base"}) for t in SAMPLE_TEXTS]
        st.session_state.loaded_files = set()
        st.session_state.pipeline_key = None

    if st.button("Clear chat"):
        st.session_state.messages = []
        st.rerun()

# Build (or rebuild) the pipeline when settings or documents change
current_key = (model_name, temperature, top_k, len(st.session_state.docs))
if st.session_state.get("pipeline_key") != current_key:
    with st.spinner("Building vector store and graph..."):
        rebuild_pipeline(model_name, temperature, top_k)

# ---------------------------------------------------------------------------
# Chat UI
# ---------------------------------------------------------------------------
st.title("🤖 Agentic RAG with LangGraph")
st.caption("Decides whether to retrieve, then answers with Groq.")

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message.get("meta"):
            render_meta(message["meta"])

submission = st.chat_input(
    "Ask a question or attach files...",
    accept_file="multiple",
    file_type=FILE_TYPES,
)

if submission:
    user_prompt = (submission.text or "").strip()
    attached = list(submission.files or [])

    # Files attached in the chat box go into the knowledge base first
    if attached:
        with st.chat_message("assistant"):
            with st.spinner("Reading files and updating the knowledge base..."):
                status = add_files(attached)
                rebuild_pipeline(model_name, temperature, top_k)
            note = "**Knowledge base updated:**\n" + "\n".join(f"- {s}" for s in status)
            st.markdown(note)
        st.session_state.messages.append({"role": "assistant", "content": note})

    if not user_prompt:
        # Only files were sent, so there is nothing to answer yet
        pass
    else:
        st.session_state.messages.append({"role": "user", "content": user_prompt})
        with st.chat_message("user"):
            st.markdown(user_prompt)

        with st.chat_message("assistant"):
            try:
                with st.spinner("Thinking..."):
                    result = st.session_state.graph.invoke(
                        {
                            "question": user_prompt,
                            "documents": [],
                            "answer": "",
                            "needs_retrieval": False,
                        }
                    )
                st.markdown(result["answer"])
                meta = {
                    "needs_retrieval": result["needs_retrieval"],
                    "documents": result["documents"],
                }
                render_meta(meta)
                st.session_state.messages.append(
                    {"role": "assistant", "content": result["answer"], "meta": meta}
                )
            except Exception as e:
                st.error(f"Something went wrong: {e}")