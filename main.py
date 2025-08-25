import os
import streamlit as st
from bs4 import BeautifulSoup
from langchain_community.document_loaders import ConfluenceLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_google_genai import GoogleGenerativeAI
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
 
# --------------------
# CONFIG
# --------------------
google_api_key = os.getenv("GOOGLE_API_KEY")
CONFLUENCE_URL = os.getenv("CONFLUENCE_URL")  # e.g., https://your-domain.atlassian.net/wiki
CONFLUENCE_USERNAME = os.getenv("CONFLUENCE_USERNAME")
CONFLUENCE_API_TOKEN = os.getenv("CONFLUENCE_API_TOKEN")
FAISS_PATH = "faiss_store"
 
# --------------------
# LOAD DATA FROM CONFLUENCE
# --------------------
def load_confluence_data(
    base_url: str,
    username: str,
    api_token: str,
    space_key: str | None = None,
    page_ids: list[str] | None = None,
    include_attachments: bool = True,
    limit: int | None = None,
):
    loader = ConfluenceLoader(
        base_url=base_url,
        username=username,
        api_key=api_token,
        space_key=space_key,
        page_ids=page_ids,
        include_attachments=include_attachments,
        limit=limit,
    )
    docs = loader.load()
 
    # Clean HTML
    cleaned_docs = []
    for doc in docs:
        soup = BeautifulSoup(doc.page_content, "html.parser")
        text = soup.get_text().strip()
        if text:
            doc.page_content = text
            cleaned_docs.append(doc)
    return cleaned_docs
 
# --------------------
# CREATE OR LOAD VECTOR DB
# --------------------
def create_vectorstore(docs):
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    docs = [doc for doc in docs if doc.page_content.strip()]
    if not docs:
        raise ValueError("No documents with text found to embed.")
 
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_documents(docs)
    if not chunks:
        raise ValueError("No chunks created from documents after splitting.")
 
    vectorstore = FAISS.from_documents(chunks, embedding_model)
    vectorstore.save_local(FAISS_PATH)
    return vectorstore
 
# --------------------
# BUILD QA CHAIN WITH MEMORY
# --------------------
def build_qa_chain(vectorstore):
    llm = GoogleGenerativeAI(
        model="gemini-2.0-flash-lite",
        google_api_key=google_api_key
    )
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    qa_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory,
    )
    return qa_chain
 
# --------------------
# STREAMLIT UI
# --------------------
st.set_page_config(page_title="Confluence Bot", layout="wide")
st.title("🤖 Confluence Chatbot")
 
# Initialize session state
for key in ["vectorstore", "qa_chain", "chat_history", "loaded_docs"]:
    if key not in st.session_state:
        st.session_state[key] = None
 
# Sidebar inputs
space_key_input = st.sidebar.text_input("Enter Confluence Space Key (required)", value="")
page_id_input = st.sidebar.text_input("Enter Confluence Page ID (optional)", value="")
include_attachments = st.sidebar.checkbox("Include Attachments", value=True)
load_docs_btn = st.sidebar.button("Load and Index Documents")
 
# Load documents and build vectorstore
if load_docs_btn:
    if not space_key_input.strip():
        st.error("Space Key is required to load documents.")
    else:
        try:
            with st.spinner("Loading documents from Confluence..."):
                page_id = page_id_input.strip()
                docs = load_confluence_data(
                    base_url=CONFLUENCE_URL,
                    username=CONFLUENCE_USERNAME,
                    api_token=CONFLUENCE_API_TOKEN,
                    space_key=space_key_input.strip(),
                    page_ids=[page_id] if page_id else None,
                    include_attachments=include_attachments
                )
 
            st.success(f"Loaded {len(docs)} document(s). Creating vectorstore...")
            vectorstore = create_vectorstore(docs)
            st.session_state.vectorstore = vectorstore
            st.session_state.qa_chain = build_qa_chain(vectorstore)
            st.session_state.chat_history = []
            st.session_state.loaded_docs = docs
 
            st.success("Vectorstore created and chatbot is ready!")
        except Exception as e:
            st.error(f"Error loading or processing documents: {e}")
            import traceback
            st.text(traceback.format_exc())
 
# Chat interface
if st.session_state.qa_chain:
    query = st.text_input("Ask a question from Confluence pages:")
    if query:
        response = st.session_state.qa_chain({"question": query})
        answer = response["answer"]
        st.session_state.chat_history.append((query, answer))
 
    # Display chat history
    for q, a in st.session_state.chat_history:
        st.markdown(f"**🧑 You:** {q}")
        st.markdown(f"**🤖 Bot:** {a}")
 
    # Optionally display loaded document titles
    if st.session_state.loaded_docs:
        with st.expander("📄 View Loaded Documents"):
            for doc in st.session_state.loaded_docs:
                st.markdown(f"- **Title:** {doc.metadata.get('title', 'Untitled')}")
else:
    st.info("Load documents first by entering Space Key (and optionally Page ID) and clicking the button above.")
