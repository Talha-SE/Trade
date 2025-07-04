from typing import List, Dict
import numpy as np
import torch
from models.tft_model import TemporalFusionTransformer  # Assuming this is the model class
from models.data_preprocessing import preprocess_data  # Assuming this function exists

class SignalGenerator:
    def __init__(self, model_path: str):
        self.model = TemporalFusionTransformer.load_from_checkpoint(model_path)
        self.model.eval()

    def generate_signals(self, input_data: np.ndarray) -> List[Dict[str, float]]:
        processed_data = preprocess_data(input_data)
        with torch.no_grad():
            predictions = self.model(processed_data)
        
        signals = self._create_signals(predictions)
        return signals

    def _create_signals(self, predictions: torch.Tensor) -> List[Dict[str, float]]:
        signals = []
        for pred in predictions:
            signal = {
                'buy_signal': float(pred[0]),  # Assuming the first element is the buy signal
                'sell_signal': float(pred[1]),  # Assuming the second element is the sell signal
                'confidence': float(pred[2])  # Assuming the third element is the confidence score
            }
            signals.append(signal)
        return signals