from fastapi import FastAPI
from app.api.routes import router
from app.config import settings
from app.model.inference import initialize_model  # This is correct and can stay

app = FastAPI(title="OpenStack LogBERT Inference")
app.include_router(router)

# ✅ Initializes model, miner, device, etc. and stores them in app.model.inference
initialize_model(settings)
