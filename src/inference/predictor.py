from fastapi import HTTPException
import torch
from src.models.tft_model import TemporalFusionTransformer
from src.utils.config import MODEL_PATH

class Predictor:
    def __init__(self):
        self.model = self.load_model()

    def load_model(self):
        try:
            model = TemporalFusionTransformer.load_from_checkpoint(MODEL_PATH)
            model.eval()
            return model
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Model loading failed: {str(e)}")

    def predict(self, input_data):
        with torch.no_grad():
            try:
                predictions = self.model(input_data)
                return predictions
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")