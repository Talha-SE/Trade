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
