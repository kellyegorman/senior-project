import os
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from sqlalchemy import create_engine, text
from sqlalchemy.exc import SQLAlchemyError

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")

if not DATABASE_URL:
    raise RuntimeError("DATABASE_URL not found in .env")

engine = create_engine(DATABASE_URL, pool_pre_ping=True)

app = FastAPI(title="Capstone API")

@app.get("/health")
def health():
    return {"ok": True}

@app.get("/db-check")
def db_check():
    try:
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        return {"db_ok": True}
    except SQLAlchemyError as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/tables")
def list_tables():
    try:
        with engine.connect() as conn:
            rows = conn.execute(text("SHOW TABLES")).fetchall()
        return {"tables": [r[0] for r in rows]}
    except SQLAlchemyError as e:
        raise HTTPException(status_code=500, detail=str(e))