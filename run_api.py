import uvicorn
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / 'src'))

if __name__ == "__main__":
    print("🚀 Starting Bitcoin Trading Bot API...")
    print("📊 API will be available at: http://localhost:8000")
    print("📚 Documentation at: http://localhost:8000/docs")
    print("🔄 Use Ctrl+C to stop")
    
    uvicorn.run(
        "src.api.signal_api:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
