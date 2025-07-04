# AI-Powered Bitcoin Trading Bot

This project implements an AI-powered Bitcoin trade signal generation system using the Temporal Fusion Transformer (TFT) model. The system is designed to analyze multi-timeframe BTC OHLCV data and generate trading signals for automated trading strategies.

## Project Structure

```
trading-bot
├── src
│   ├── main.py                # Entry point for the FastAPI application
│   ├── api
│   │   ├── __init__.py        # Marks the api directory as a package
│   │   └── routes.py          # Defines API routes for trading signals
│   ├── models
│   │   ├── __init__.py        # Marks the models directory as a package
│   │   ├── tft_model.py       # Implementation of the Temporal Fusion Transformer model
│   │   └── data_preprocessing.py # Functions for preprocessing OHLCV data
│   ├── data
│   │   ├── __init__.py        # Marks the data directory as a package
│   │   ├── collector.py        # Functions for collecting historical data using ccxt
│   │   └── ccxt_client.py     # Initializes the ccxt client for exchanges
│   ├── training
│   │   ├── __init__.py        # Marks the training directory as a package
│   │   ├── train_pipeline.py   # Training pipeline for the model
│   │   └── model_utils.py      # Utility functions for model evaluation
│   ├── inference
│   │   ├── __init__.py        # Marks the inference directory as a package
│   │   ├── signal_generator.py  # Logic for generating trading signals
│   │   └── predictor.py        # Handles inference and predictions
│   └── utils
│       ├── __init__.py        # Marks the utils directory as a package
│       └── config.py          # Configuration settings and constants
├── notebooks
│   ├── data_exploration.ipynb  # Jupyter notebook for data exploration
│   ├── model_training.ipynb     # Jupyter notebook for model training
│   └── backtesting.ipynb        # Jupyter notebook for backtesting strategies
├── requirements.txt             # Lists project dependencies
├── .env.example                  # Template for environment variables
└── README.md                    # Project documentation
```

## Setup Instructions

1. Clone the repository:
   ```
   git clone https://github.com/yourusername/trading-bot.git
   cd trading-bot
   ```

2. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Set up your environment variables by copying `.env.example` to `.env` and updating the values as needed.

## Usage

- Start the FastAPI application:
  ```
  uvicorn src.main:app --reload
  ```

- Access the API documentation at `http://127.0.0.1:8000/docs`.

## Overview

This trading bot leverages advanced machine learning techniques to analyze Bitcoin price data and generate actionable trading signals. The modular design allows for easy updates and scalability, making it suitable for both individual traders and institutional use. 

## Contributing

Contributions are welcome! Please open an issue or submit a pull request for any enhancements or bug fixes.