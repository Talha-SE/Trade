import os

class Config:
    # General settings
    PROJECT_NAME = "AI-Powered Bitcoin Trading Bot"
    VERSION = "1.0.0"

    # Data settings
    DATA_PATH = os.getenv("DATA_PATH", "./data")
    OHLCV_DATA_FILE = os.path.join(DATA_PATH, "ohlcv_data.csv")

    # Model settings
    MODEL_PATH = os.getenv("MODEL_PATH", "./models")
    TFT_MODEL_FILE = os.path.join(MODEL_PATH, "tft_model.pth")

    # Training settings
    NUM_EPOCHS = int(os.getenv("TRAINING_EPOCHS", 100))
    BATCH_SIZE = int(os.getenv("BATCH_SIZE", 32))
    LEARNING_RATE = float(os.getenv("LEARNING_RATE", 0.001))

    # Google Drive settings
    GDRIVE_MODEL_PATH = os.getenv("GDRIVE_MODEL_PATH", "/content/drive/MyDrive/trading_bot/models")
    GDRIVE_DATA_PATH = os.getenv("GDRIVE_DATA_PATH", "/content/drive/MyDrive/trading_bot/data")

    # CCXT settings
    CCXT_EXCHANGE = os.getenv("CCXT_EXCHANGE", "binance")
    CCXT_API_KEY = os.getenv("CCXT_API_KEY")
    CCXT_SECRET = os.getenv("CCXT_SECRET")

    # API settings
    API_HOST = os.getenv("API_HOST", "0.0.0.0")
    API_PORT = int(os.getenv("API_PORT", 8000))

    # Trading settings
    TRADING_SYMBOL = "BTC/USDT"
    TIMEFRAMES = ["1m", "5m", "15m", "30m", "1h", "4h", "1d"]
    
    # Data collection periods
    HIGH_FREQ_PERIOD = 6 * 30 * 24 * 60  # 6 months in minutes for 1m, 5m, 15m, 30m, 1h
    LOW_FREQ_PERIOD = 365 * 24  # 1 year in hours for 4h, 1d

    # Other settings
    LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
    DEBUG_MODE = os.getenv("DEBUG_MODE", "False").lower() == "true"