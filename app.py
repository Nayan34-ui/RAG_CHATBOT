import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import HuggingFaceBgeEmbeddings
from langchain.vectorstores import FAISS
import tempfile
import os

st.set_page_config(
    page_title="RAG Document Chatbot",
    page_icon="🤖",
    layout="centered"
)

st.title("🤖 RAG Document Chatbot")
st.caption("Upload a PDF and ask questions from it — powered by HuggingFace Embeddings + FAISS")

with st.sidebar:
    st.header("⚙️ How to use")
    st.markdown("""
    1. Upload a PDF file
    2. Click **Build Knowledge Base**
    3. Ask any question about the document!
    """)
    st.markdown("---")
    st.markdown("Built by **Nayan Kadu**")
    st.markdown("[GitHub](https://github.com/Nayan34-ui) · [LinkedIn](https://linkedin.com/in/nayan-kadu-655b39282)")

if "vector_db" not in st.session_state:
    st.session_state.vector_db = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

uploaded_file = st.file_uploader("Upload a PDF file", type=["pdf"])

if st.button("🔨 Build Knowledge Base", type="primary"):
    if uploaded_file is None:
        st.error("Please upload a PDF file first.")
    else:
        with st.spinner("Loading and indexing your PDF..."):
            try:
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                    tmp_file.write(uploaded_file.read())
                    tmp_path = tmp_file.name

                loader = PyPDFLoader(tmp_path)
                documents = loader.load()

                splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
                chunks = splitter.split_documents(documents)

                embedding_model = HuggingFaceBgeEmbeddings(model_name="all-MiniLM-L6-v2")
                vector_db = FAISS.from_documents(chunks, embedding_model)
                st.session_state.vector_db = vector_db
                st.session_state.chat_history = []

                os.unlink(tmp_path)
                st.success(f"✅ Knowledge base built from {len(chunks)} chunks! Now ask your questions below.")

            except Exception as e:
                st.error(f"Error building knowledge base: {e}")

if st.session_state.vector_db:
    st.markdown("---")
    st.subheader("💬 Ask a Question")

    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])

    question = st.chat_input("Type your question here...")

    if question:
        st.session_state.chat_history.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.write(question)

        with st.chat_message("assistant"):
            with st.spinner("Searching document..."):
                try:
                    relevant_docs = st.session_state.vector_db.similarity_search(question, k=3)
                    answer = ""
                    for i, doc in enumerate(relevant_docs, 1):
                        answer += f"**Chunk {i}:**\n{doc.page_content}\n\n"
                    st.markdown(answer)
                    st.session_state.chat_history.append({"role": "assistant", "content": answer})
                except Exception as e:
                    st.error(f"Error searching: {e}")

elif uploaded_file is None:
    st.info("👆 Upload a PDF and click **Build Knowledge Base** to get started.")
