import os
from dotenv import load_dotenv
from sqlalchemy import create_engine, text
from dotenv import load_dotenv

# Load your environment variables (.env file)
load_dotenv()
DB_URI = os.getenv("DATABASE_URL")

# Create the database engine
engine = create_engine(DB_URI)

# Execute the command to enable the vector extension
with engine.connect() as conn:
  conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector;"))
  conn.commit()
  print("Success: Vector extension enabled in RDS PostgreSQL!")