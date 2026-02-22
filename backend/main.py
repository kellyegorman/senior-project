import os
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Header
from sqlalchemy import create_engine, text
from sqlalchemy.exc import SQLAlchemyError
from pydantic import BaseModel, Field
import uuid
from passlib.context import CryptContext
from dotenv import load_dotenv

load_dotenv()
API_KEY = os.getenv("API_KEY")

def require_api_key(x_api_key: str | None):
    if not API_KEY:
        raise RuntimeError("API_KEY not set")
    if x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key")

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")

if not DATABASE_URL:
    raise RuntimeError("DATABASE_URL not found in .env")

engine = create_engine(DATABASE_URL, pool_pre_ping=True)

app = FastAPI(title="Capstone API")

class UserCreate(BaseModel):
    username: str
    email: str
    password: str = Field(..., min_length=8, max_length=72)

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

@app.post("/users")
def create_user(user: UserCreate, x_api_key: str | None = Header(default=None)):
    require_api_key(x_api_key)
    try:
        user_id = str(uuid.uuid4())[:15]
        hashed_password = pwd_context.hash(user.password)

        with engine.connect() as conn:
            conn.execute(
                text("""
                    INSERT INTO users (userid, username, email, password_hash)
                    VALUES (:userid, :username, :email, :password)
                """),
                {
                    "userid": user_id,
                    "username": user.username,
                    "email": user.email,
                    "password": hashed_password 
                }
            )
            conn.commit()

        return {"message": "User created", "userid": user_id}

    except SQLAlchemyError as e:
        raise HTTPException(status_code=500, detail=str(e))
    
@app.get("/users")
def get_users():
    with engine.connect() as conn:
        rows = conn.execute(text("SELECT userid, username, email FROM users")).fetchall()
    return {"users": [dict(r._mapping) for r in rows]}