from typing import Tuple
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

def preprocess_ohlcv(data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Preprocess the OHLCV data for model training.
    
    Args:
        data (pd.DataFrame): Raw OHLCV data.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: Scaled features and target DataFrames.
    """
    # Ensure the data is sorted by date
    data = data.sort_values('date')

    # Extract features and target
    features = data[['open', 'high', 'low', 'close', 'volume']]
    target = data['close'].shift(-1)  # Predict the next close price

    # Scale features
    scaler = MinMaxScaler()
    scaled_features = scaler.fit_transform(features)

    # Create DataFrame for scaled features
    scaled_features_df = pd.DataFrame(scaled_features, columns=features.columns)

    return scaled_features_df, target.dropna()  # Drop the last row of target as it will be NaN after shift

def prepare_tft_data(data_dict, target_timeframe='1h', config=None):
    """Prepare data for TFT training with proper data types"""
    
    # Use 1-hour data as primary timeframe
    main_data = data_dict[config.TRADING_SYMBOL][target_timeframe].copy()
    main_data['timestamp'] = pd.to_datetime(main_data['timestamp'], unit='ms')
    main_data = main_data.sort_values('timestamp').reset_index(drop=True)
    
    # Create features
    main_data['returns'] = main_data['close'].pct_change()
    main_data['high_low_ratio'] = main_data['high'] / main_data['low']
    main_data['close_open_ratio'] = main_data['close'] / main_data['open']
    main_data['volume_ma'] = main_data['volume'].rolling(20).mean()
    
    # Technical indicators
    main_data['sma_20'] = main_data['close'].rolling(20).mean()
    main_data['sma_50'] = main_data['close'].rolling(50).mean()
    
    # RSI calculation
    delta = main_data['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    main_data['rsi'] = 100 - (100 / (1 + rs))
    
    # Create target (next hour's return)
    main_data['target'] = main_data['returns'].shift(-1)
    
    # Add time features for TFT
    main_data['hour'] = main_data['timestamp'].dt.hour
    main_data['day_of_week'] = main_data['timestamp'].dt.dayofweek
    main_data['month'] = main_data['timestamp'].dt.month
    
    # Add time index and group (IMPORTANT: Convert to string)
    main_data['time_idx'] = range(len(main_data))
    main_data['group_id'] = 'BTC'  # String type, not numeric
    
    # Convert categorical columns to string type
    main_data['hour'] = main_data['hour'].astype(str)
    main_data['day_of_week'] = main_data['day_of_week'].astype(str)
    main_data['month'] = main_data['month'].astype(str)
    main_data['group_id'] = main_data['group_id'].astype(str)
    
    # Remove NaN values
    main_data = main_data.dropna()
    
    # Ensure we have enough data
    if len(main_data) < 100:
        raise ValueError(f"Insufficient data after preprocessing: {len(main_data)} records. Need at least 100.")
    
    return main_data
