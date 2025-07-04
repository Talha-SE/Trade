from pathlib import Path
import pandas as pd
import torch
from torch.utils.data import DataLoader
from models.data_preprocessing import preprocess_ohlcv
from models.tft_model import TFTModel
from utils.config import Config

def load_data(file_path):
    data = pd.read_csv(file_path)
    return preprocess_ohlcv(data)

def train_model(train_loader, model, criterion, optimizer, num_epochs):
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for batch_idx, (inputs, targets) in enumerate(train_loader):
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        avg_loss = total_loss / len(train_loader)
        print(f'Epoch {epoch+1}/{num_epochs}, Average Loss: {avg_loss:.6f}')

def save_model(model, model_path):
    Path(model_path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), model_path)

def main():
    config = Config()
    data_file = config.OHLCV_DATA_FILE
    model_path = config.TFT_MODEL_FILE
    num_epochs = config.NUM_EPOCHS
    batch_size = config.BATCH_SIZE

    # Create directories if they don't exist
    Path(config.DATA_PATH).mkdir(parents=True, exist_ok=True)
    Path(config.MODEL_PATH).mkdir(parents=True, exist_ok=True)

    # Load and preprocess data
    features, targets = load_data(data_file)
    
    # Create dataset and dataloader
    from torch.utils.data import TensorDataset
    dataset = TensorDataset(torch.FloatTensor(features.values), torch.FloatTensor(targets.values))
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Initialize model
    model = TFTModel(
        input_size=features.shape[1],
        output_size=1,
        hidden_size=64,
        num_layers=2,
        dropout=0.1
    )
    
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.LEARNING_RATE)

    train_model(train_loader, model, criterion, optimizer, num_epochs)
    save_model(model, model_path)

if __name__ == "__main__":
    main()