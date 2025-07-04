from fastapi import APIRouter
from src.inference.signal_generator import generate_signals

router = APIRouter()

@router.get("/signals")
async def get_signals():
    signals = generate_signals()
    return {"signals": signals}

@router.get("/health")
async def health_check():
    return {"status": "healthy"}