import os
from dotenv import load_dotenv
import streamlit as st
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_postgres import PGVector
from langchain_text_splitters import RecursiveCharacterTextSplitter

# 1. Load environment variables for local development (.env file)
load_dotenv()

# Retrieve credentials securely (supports both .env and Streamlit secrets safely)
DB_URI = os.getenv("DB_URI")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

try:
  if not DB_URI and "DB_URI" in st.secrets:
    DB_URI = st.secrets["DB_URI"]
  if not OPENAI_API_KEY and "OPENAI_API_KEY" in st.secrets:
    OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]
except Exception:
  pass

# 2. Configure Streamlit Page
st.set_page_config(
    page_title="RAG Assistant with AWS RDS PostgreSQL & OpenRouter",
    layout="wide",
)
st.title("🧠 RAG Assistant with AWS RDS PostgreSQL")

# Validation check
if not DB_URI or not OPENAI_API_KEY:
  st.error(
      "Missing configuration! Please check that DB_URI and OPENAI_API_KEY are"
      " set in your .env file or Streamlit secrets."
  )
  st.stop()


# 3. Initialize Embeddings and Vector Store via OpenRouter
@st.cache_resource
def init_vector_store():
  embeddings = OpenAIEmbeddings(
      model="openai/text-embedding-3-small",
      openai_api_key=OPENAI_API_KEY,
      base_url="https://openrouter.ai/api/v1",
      check_embedding_ctx_length=False,
  )
  vector_store = PGVector(
      embeddings=embeddings,
      collection_name="rag_documents",
      connection=DB_URI,
      use_jsonb=True,
  )
  return vector_store


vector_store = init_vector_store()
retriever = vector_store.as_retriever(
    search_type="similarity", search_kwargs={"k": 3}
)

# 4. Sidebar: Add Documents to RDS PostgreSQL
st.sidebar.header("📁 Knowledge Base Manager")
uploaded_text = st.sidebar.text_area(
    "Paste text chunks to add to your database:"
)

if st.sidebar.button("Add to RDS Database"):
  if uploaded_text:
    with st.spinner("Processing and storing embeddings..."):
      text_splitter = RecursiveCharacterTextSplitter(
          chunk_size=500, chunk_overlap=50
      )
      docs = text_splitter.create_documents([uploaded_text])
      vector_store.add_documents(docs)
      st.sidebar.success(
          f"Successfully stored {len(docs)} chunks into AWS RDS!"
      )
  else:
    st.sidebar.warning("Please enter text before submitting.")

# 5. Main Interface: RAG Query & Chat
st.subheader("💬 Ask Your Documents")
query = st.text_input("Enter your question:")

if query:
  with st.spinner("Searching AWS RDS and generating answer..."):
    # Retrieve relevant document chunks from PostgreSQL
    relevant_docs = retriever.invoke(query)

    # Initialize LLM via OpenRouter
    llm = ChatOpenAI(
        model="openai/gpt-4o-mini",
        temperature=0,
        openai_api_key=OPENAI_API_KEY,
        base_url="https://openrouter.ai/api/v1",
    )

    # Construct Prompt
    template = """Answer the question based only on the following context:
    {context}

    Question: {question}
    """
    prompt = ChatPromptTemplate.from_template(template)


    def format_docs(docs):
      return "\n\n".join(doc.page_content for doc in docs)


    context_text = format_docs(relevant_docs)
    formatted_prompt = prompt.format(context=context_text, question=query)

    # Generate response
    response = llm.invoke(formatted_prompt)

    # Display Answer
    st.write("### Answer:")
    st.write(response.content)

    # Display Source Context in an Expander
    with st.expander("View Retrieved Context Chunks from RDS"):
      for i, doc in enumerate(relevant_docs):
        st.markdown(f"**Chunk {i+1}:**")
        st.write(doc.page_content)