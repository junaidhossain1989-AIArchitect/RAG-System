import os
import math
from openai import OpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter

print("1. Script started successfully...")

# 1. Initialize OpenAI Client configured for OpenRouter
client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.environ.get("OPENAI_API_KEY"),
)

# 2. Your Private Knowledge Data (The "Book")
knowledge_base_text = """
Project Orion is a secret hobby initiative started by Alex in 2026. 
The goal is to build an automated tomato watering system using a Raspberry Pi 5.
The project budget is exactly $150. It uses a soil moisture sensor model SMS-X1.
Alex keeps the prototype in the backyard shed behind the lawnmower."""

# 3. Chunk the Data
text_splitter = RecursiveCharacterTextSplitter(chunk_size=150, chunk_overlap=20)
chunks = text_splitter.split_text(knowledge_base_text)
print(f"2. Text split into {len(chunks)} chunks.")

# 4. Generate Embeddings for Chunks via OpenRouter
print("3. Generating embeddings via OpenRouter...")
response_embeddings = client.embeddings.create(
    model="openai/text-embedding-3-small",
    input=chunks
)
chunk_embeddings = [item.embedding for item in response_embeddings.data]
print("4. Embeddings generated successfully!")

# 5. Ask a Question
user_question = "Where is the prototype stored and what is the budget?"
print(f"5. Querying for: '{user_question}'")

# 6. Generate query embedding
query_response = client.embeddings.create(
    model="openai/text-embedding-3-small",
    input=[user_question]
)
query_embedding = query_response.data[0].embedding

# 7. Pure Python Cosine Similarity Search
def cosine_similarity(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    mag_a = math.sqrt(sum(x * x for x in a))
    mag_b = math.sqrt(sum(y * y for y in b))
    if mag_a == 0 or mag_b == 0:
        return 0.0
    return dot / (mag_a * mag_b)

# Score each chunk against the query
scored_chunks = []
for i, chunk_emb in enumerate(chunk_embeddings):
    score = cosine_similarity(query_embedding, chunk_emb)
    scored_chunks.append((score, chunks[i]))

# Sort by highest similarity and pick the top 2
scored_chunks.sort(key=lambda x: x[0], reverse=True)
retrieved_context = " ".join([chunk for score, chunk in scored_chunks[:2]])

# 8. Augment & Generate via OpenRouter Chat
print("6. Generating final answer from LLM...")
response = client.chat.completions.create(
    model="openai/gpt-4o-mini",
    messages=[
        {"role": "system", "content": "You are a helpful assistant. Answer the user's question using ONLY the provided Context. If the context doesn't contain the answer, say 'I don't know'."},
        {"role": "user", "content": f"Context: {retrieved_context}\n\nQuestion: {user_question}"}
    ]
)

# Print the final accurate answer
print("\n--- AI ANSWER ---")
print(response.choices[0].message.content)