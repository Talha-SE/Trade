from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import torch
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
import sys
import os

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from utils.config import Config
from data.collector import collect_data

app = FastAPI(title="Bitcoin Trading Bot API", version="1.0.0")
config = Config()

# Global variables for model and data
model = None
last_data_update = None
cached_data = None

class SignalResponse(BaseModel):
    signal: str
    confidence: float
    predicted_return: float
    current_price: float
    timestamp: str
    technical_indicators: dict

class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    last_update: str
    data_points: int

def load_trained_model():
    """Load the trained TFT model"""
    global model
    
    model_files = [
        config.MODEL_PATH + '/best_tft_model.pth',
        config.MODEL_PATH + '/tft_model_state.pth',
        config.MODEL_PATH + '/tft_model_full.pth'
    ]
    
    for model_file in model_files:
        try:
            if os.path.exists(model_file):
                if model_file.endswith('_full.pth'):
                    model = torch.load(model_file, map_location='cpu')
                    print(f"✅ Model loaded from: {model_file}")
                    return True
                else:
                    # For state dict, we'd need to recreate the model architecture
                    print(f"⚠️  State dict found but need model architecture: {model_file}")
            else:
                print(f"⚠️  Model file not found: {model_file}")
        except Exception as e:
            print(f"❌ Failed to load {model_file}: {e}")
    
    print("⚠️  No model loaded, using rule-based signals")
    return False

def get_latest_data():
    """Fetch latest Bitcoin data"""
    global cached_data, last_data_update
    
    try:
        # Check if we need to update data (every 5 minutes)
        if (last_data_update is None or 
            datetime.now() - last_data_update > timedelta(minutes=5)):
            
            print("🔄 Fetching latest data...")
            symbols = [config.TRADING_SYMBOL]
            timeframes = ["1h"]
            since = int((datetime.now() - timedelta(hours=100)).timestamp() * 1000)
            
            data = collect_data(symbols, timeframes, since, limit=100)
            cached_data = data[config.TRADING_SYMBOL]["1h"]
            last_data_update = datetime.now()
            
        return cached_data
        
    except Exception as e:
        print(f"❌ Data collection failed: {e}")
        # Return dummy data
        return pd.DataFrame({
            'timestamp': [int(datetime.now().timestamp() * 1000)],
            'open': [45000], 'high': [45500], 'low': [44500], 
            'close': [45200], 'volume': [1000000]
        })

def calculate_technical_indicators(data):
    """Calculate technical indicators"""
    data = data.copy()
    
    # Basic indicators
    data['sma_20'] = data['close'].rolling(20).mean()
    data['sma_50'] = data['close'].rolling(50).mean()
    data['returns'] = data['close'].pct_change()
    
    # RSI
    delta = data['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    data['rsi'] = 100 - (100 / (1 + rs))
    
    # Bollinger Bands
    data['bb_middle'] = data['close'].rolling(20).mean()
    bb_std = data['close'].rolling(20).std()
    data['bb_upper'] = data['bb_middle'] + (bb_std * 2)
    data['bb_lower'] = data['bb_middle'] - (bb_std * 2)
    
    return data

def generate_signal(data):
    """Generate trading signal"""
    if len(data) < 50:
        return "HOLD", 0.5, 0.0
    
    latest = data.iloc[-1]
    
    # Multi-factor signal generation
    signals = []
    confidences = []
    
    # RSI Signal
    if pd.notna(latest['rsi']):
        if latest['rsi'] < 30:
            signals.append("BUY")
            confidences.append(0.7)
        elif latest['rsi'] > 70:
            signals.append("SELL")
            confidences.append(0.7)
        else:
            signals.append("HOLD")
            confidences.append(0.3)
    
    # Moving Average Signal
    if pd.notna(latest['sma_20']) and pd.notna(latest['sma_50']):
        if latest['sma_20'] > latest['sma_50']:
            signals.append("BUY")
            confidences.append(0.6)
        else:
            signals.append("SELL")
            confidences.append(0.6)
    
    # Bollinger Bands Signal
    if pd.notna(latest['bb_upper']) and pd.notna(latest['bb_lower']):
        if latest['close'] < latest['bb_lower']:
            signals.append("BUY")
            confidences.append(0.8)
        elif latest['close'] > latest['bb_upper']:
            signals.append("SELL")
            confidences.append(0.8)
    
    # Momentum Signal
    if len(data) >= 10:
        momentum = data['returns'].tail(5).mean()
        if momentum > 0.01:
            signals.append("BUY")
            confidences.append(0.5)
        elif momentum < -0.01:
            signals.append("SELL")
            confidences.append(0.5)
    
    # Aggregate signals
    if not signals:
        return "HOLD", 0.5, 0.0
    
    # Count votes
    buy_votes = signals.count("BUY")
    sell_votes = signals.count("SELL")
    
    if buy_votes > sell_votes:
        final_signal = "BUY"
        confidence = np.mean([c for s, c in zip(signals, confidences) if s == "BUY"])
    elif sell_votes > buy_votes:
        final_signal = "SELL"
        confidence = np.mean([c for s, c in zip(signals, confidences) if s == "SELL"])
    else:
        final_signal = "HOLD"
        confidence = 0.5
    
    # Predicted return (simple estimate)
    predicted_return = (confidence - 0.5) * 0.05 if final_signal == "BUY" else -(confidence - 0.5) * 0.05
    
    return final_signal, confidence, predicted_return

@app.on_event("startup")
async def startup_event():
    """Initialize the API"""
    print("🚀 Starting Bitcoin Trading Bot API...")
    load_trained_model()
    print("✅ API Ready!")

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    global model, last_data_update, cached_data
    
    return HealthResponse(
        status="healthy",
        model_loaded=model is not None,
        last_update=last_data_update.isoformat() if last_data_update else "never",
        data_points=len(cached_data) if cached_data is not None else 0
    )

@app.get("/signal", response_model=SignalResponse)
async def get_trading_signal():
    """Get current trading signal"""
    try:
        # Get latest data
        data = get_latest_data()
        
        if data is None or len(data) == 0:
            raise HTTPException(status_code=500, detail="Unable to fetch market data")
        
        # Calculate technical indicators
        data_with_indicators = calculate_technical_indicators(data)
        
        # Generate signal
        signal, confidence, predicted_return = generate_signal(data_with_indicators)
        
        # Get current price
        current_price = float(data_with_indicators.iloc[-1]['close'])
        
        # Get latest technical indicators
        latest = data_with_indicators.iloc[-1]
        technical_indicators = {
            "rsi": float(latest['rsi']) if pd.notna(latest['rsi']) else None,
            "sma_20": float(latest['sma_20']) if pd.notna(latest['sma_20']) else None,
            "sma_50": float(latest['sma_50']) if pd.notna(latest['sma_50']) else None,
            "bb_upper": float(latest['bb_upper']) if pd.notna(latest['bb_upper']) else None,
            "bb_lower": float(latest['bb_lower']) if pd.notna(latest['bb_lower']) else None,
        }
        
        return SignalResponse(
            signal=signal,
            confidence=confidence,
            predicted_return=predicted_return,
            current_price=current_price,
            timestamp=datetime.now().isoformat(),
            technical_indicators=technical_indicators
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating signal: {str(e)}")

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Bitcoin Trading Bot API",
        "version": "1.0.0",
        "endpoints": {
            "/health": "Health check",
            "/signal": "Get trading signal",
            "/docs": "API documentation"
        }
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
