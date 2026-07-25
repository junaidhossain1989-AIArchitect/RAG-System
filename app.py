import os
import requests
import chromadb
import streamlit as st
from chromadb.utils.embedding_functions import OllamaEmbeddingFunction

# --- 1. PAGE CONFIGURATION ---
st.set_page_config(page_title="Local RAG Chatbot", page_icon="🤖", layout="centered")
st.title("🤖 My Private Knowledge Base")
st.caption("A 100% free, local RAG system running on your laptop via Ollama.")

# --- 2. INITIALISE DATABASE & EMBEDDINGS ---
@st.cache_resource
def init_rag():
    """Connects to the database once and keeps it in cache for speed."""
    db_path = os.path.join(os.getcwd(), "my_local_db")
    client = chromadb.PersistentClient(path=db_path)
    
    ollama_embedding = OllamaEmbeddingFunction(
        url="http://localhost:11434/api/embeddings",
        model_name="nomic-embed-text"
    )
    
    collection = client.get_or_create_collection(
        name="my_free_knowledge_base", 
        embedding_function=ollama_embedding
    )
    return collection

collection = init_rag()

# --- 3. SIDEBAR: FILE UPLOADER & STATS ---
with st.sidebar:
    st.header("📂 Document Management")
    st.markdown("Upload fresh text or markdown files directly into your local vector database.")
    
    uploaded_file = st.file_uploader("Choose a file", type=["txt", "md"])
    
    if uploaded_file is not None:
        file_contents = uploaded_file.read().decode("utf-8")
        file_name = uploaded_file.name
        
        if st.button("Ingest File", type="primary"):
            with st.spinner(f"Embedding and storing {file_name}..."):
                try:
                    # Upsert ensures that re-uploading an updated file overwrites the old version
                    collection.upsert(
                        documents=[file_contents],
                        ids=[file_name],
                        metadatas=[{"source": file_name}]
                    )
                    st.success(f"Successfully added **{file_name}**!")
                except Exception as e:
                    st.error(f"Error adding file: {e}")

    st.markdown("---")
    st.markdown("### 📊 Database Status")
    try:
        doc_count = collection.count()
        st.metric(label="Stored Documents", value=doc_count)
    except Exception:
        st.metric(label="Stored Documents", value="Unavailable")

# --- 4. CHAT HISTORY MANAGEMENT ---
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Ask me anything about your loaded documents, or upload a new file in the sidebar!"}
    ]

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

# --- 5. HANDLE USER INPUT ---
if user_question := st.chat_input("Type your question here..."):
    
    with st.chat_message("user"):
        st.write(user_question)
    st.session_state.messages.append({"role": "user", "content": user_question})

    with st.chat_message("assistant"):
        with st.spinner("Searching local files and thinking..."):
            
            # A. Retrieve relevant text chunks
            results = collection.query(query_texts=[user_question], n_results=1)
            retrieved_context = results['documents'][0] if results['documents'] and results['documents'][0] else "No relevant context found."
            
            # B. Build the prompt for Llama 3.2
            system_prompt = f"""
            You are a helpful assistant. Answer the user's question using ONLY the provided Context. 
            If the context doesn't contain the answer, say 'I don't know'.

            Context: {retrieved_context}
            Question: {user_question}
            """
            
            # C. Send to local Ollama API
            try:
                response = requests.post(
                    "http://localhost:11434/api/generate",
                    json={
                        "model": "llama3.2",
                        "prompt": system_prompt,
                        "stream": False
                    },
                    timeout=30
                )
                ai_answer = response.json()['response']
            except Exception as e:
                ai_answer = f"⚠️ Error connecting to Ollama: {str(e)}. Make sure Ollama is running in your terminal."

            st.write(ai_answer)
            
            with st.expander("📚 View retrieved source context used for this answer"):
                st.info(retrieved_context)

    st.session_state.messages.append({"role": "assistant", "content": ai_answer})