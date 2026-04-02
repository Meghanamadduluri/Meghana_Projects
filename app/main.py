import os
from pathlib import Path

from dotenv import load_dotenv

# When you run `uvicorn` from the repo root, cwd is not always `app/`, so load both.
_here = Path(__file__).resolve().parent
load_dotenv(_here / ".env")
load_dotenv(_here.parent / ".env")
from fastapi import FastAPI 
from app.api.routes import router

app = FastAPI() 

app.include_router(router)




