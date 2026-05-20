# 🤖 RAG Document Chatbot

A Retrieval-Augmented Generation (RAG) based AI chatbot that answers questions from any PDF document using HuggingFace Embeddings, FAISS vector search, and Groq LLM.

## 🚀 Live Demo
👉 [nayan-rag-chatbot.streamlit.app](https://nayan-rag-chatbot.streamlit.app)

## 🛠️ Tech Stack
- **LLM:** Groq (Llama 3.3 70B)
- **Embeddings:** HuggingFace (all-MiniLM-L6-v2)
- **Vector Store:** FAISS
- **Framework:** LangChain
- **UI:** Streamlit

## ⚙️ How It Works
1. Upload any PDF document
2. Document is split into chunks and indexed using FAISS
3. User query is matched with relevant chunks via semantic search
4. Groq LLM generates a clean, context-aware answer

## 📦 Run Locally
```bash
pip install -r requirements.txt
streamlit run app.py
```

## 👨‍💻 Built By
**Nayan Kadu** — [GitHub](https://github.com/Nayan34-ui) · [LinkedIn](https://linkedin.com/in/nayan-kadu-655b39282)
