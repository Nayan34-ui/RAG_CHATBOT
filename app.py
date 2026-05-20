import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from groq import Groq
import tempfile
import os

st.set_page_config(
    page_title="RAG Document Chatbot",
    page_icon="🤖",
    layout="centered"
)

st.title("🤖 RAG Document Chatbot")
st.caption("Upload a PDF and ask questions — powered by HuggingFace + FAISS + Groq")

with st.sidebar:
    st.header("⚙️ How to use")
    st.markdown("""
    1. Upload a PDF file
    2. Click **Build Knowledge Base**
    3. Ask any question!
    """)
    st.markdown("---")
    st.markdown("Built by **Nayan Kadu**")
    st.markdown("[GitHub](https://github.com/Nayan34-ui) · [LinkedIn](https://linkedin.com/in/nayan-kadu-655b39282)")

if "vector_db" not in st.session_state:
    st.session_state.vector_db = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

GROQ_API_KEY = st.secrets["GROQ_API_KEY"]

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

                embedding_model = HuggingFaceEmbeddings(model="llama-3.3-70b-versatile")
                vector_db = FAISS.from_documents(chunks, embedding_model)
                st.session_state.vector_db = vector_db
                st.session_state.chat_history = []
                os.unlink(tmp_path)
                st.success(f"✅ Knowledge base ready! {len(chunks)} chunks indexed.")

            except Exception as e:
                st.error(f"Error: {e}")

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
            with st.spinner("Thinking..."):
                try:
                    relevant_docs = st.session_state.vector_db.similarity_search(question, k=3)
                    context = "\n\n".join([doc.page_content for doc in relevant_docs])

                    client = Groq(api_key=GROQ_API_KEY)
                    response = client.chat.completions.create(
                        model="llama3-8b-8192",
                        messages=[
                            {"role": "system", "content": "You are a helpful assistant. Answer the question based only on the provided context. Be concise and clear."},
                            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"}
                        ]
                    )
                    answer = response.choices[0].message.content
                    st.write(answer)
                    st.session_state.chat_history.append({"role": "assistant", "content": answer})

                except Exception as e:
                    st.error(f"Error: {e}")

elif uploaded_file is None:
    st.info("👆 Upload a PDF and click **Build Knowledge Base** to get started.")
