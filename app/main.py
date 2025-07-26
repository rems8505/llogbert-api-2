from fastapi import FastAPI
from app.api.routes import router
from app.config import settings
from app.model.inference import initialize_model

app = FastAPI(title="OpenStack LogBERT Inference")
app.include_router(router)

# Initialize model & miner
initialize_model(settings)
