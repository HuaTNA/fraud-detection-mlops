from fastapi import FastAPI
from api.routers import fraud

app = FastAPI()
app.include_router(fraud.router, prefix="/predict")
