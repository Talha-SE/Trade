from pytorch_forecasting import TemporalFusionTransformer, TimeSeriesDataSet
import torch
import pandas as pd

class TFTModel:
    def __init__(self, input_size, output_size, hidden_size, num_layers, dropout):
        self.model = TemporalFusionTransformer(
            input_size=input_size,
            output_size=output_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout
        )

    def train(self, train_data: pd.DataFrame, epochs: int, batch_size: int):
        train_dataset = TimeSeriesDataSet(
            train_data,
            time_idx="time_idx",
            target="target",
            group_ids=["group_id"],
            min_encoder_length=1,
            max_encoder_length=24,
            min_prediction_length=1,
            max_prediction_length=6,
            static_categoricals=["static_cat"],
            static_reals=["static_real"],
            time_varying_known_categoricals=["known_cat"],
            time_varying_known_reals=["known_real"],
            time_varying_unknown_categoricals=["unknown_cat"],
            time_varying_unknown_reals=["unknown_real"],
        )

        train_dataloader = train_dataset.to_dataloader(train=True, batch_size=batch_size)

        trainer = pl.Trainer(max_epochs=epochs)
        trainer.fit(self.model, train_dataloader)

    def predict(self, data: pd.DataFrame):
        self.model.eval()
        with torch.no_grad():
            predictions = self.model(data)
        return predictions

    def save_model(self, path: str):
        torch.save(self.model.state_dict(), path)

    def load_model(self, path: str):
        self.model.load_state_dict(torch.load(path))