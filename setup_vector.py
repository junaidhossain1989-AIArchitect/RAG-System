import os
from dotenv import load_dotenv
from sqlalchemy import create_engine, text

# Load your environment variables (.env file)
load_dotenv()
DB_URI = os.getenv("DB_URI")

if not DB_URI:
    raise ValueError("ERROR: DATABASE_URL is not set in your environment or .env file.")

print("Connecting to database...")

try:
    # Create the database engine
    engine = create_engine(DB_URI)

    # Execute the command to enable the vector extension within an automated transaction block
    with engine.begin() as conn:
        conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector;"))
        
    print("Success: Vector extension enabled in RDS PostgreSQL!")

except Exception as e:
    print(f"CRITICAL ERROR: Failed to enable vector extension. Details: {e}")