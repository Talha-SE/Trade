def calculate_metrics(predictions, targets):
    mse = ((predictions - targets) ** 2).mean()  # Mean Squared Error
    rmse = mse ** 0.5  # Root Mean Squared Error
    mae = abs(predictions - targets).mean()  # Mean Absolute Error
    return {"mse": mse, "rmse": rmse, "mae": mae}

def save_model(model, filepath):
    import torch
    torch.save(model.state_dict(), filepath)

def load_model(model_class, filepath):
    import torch
    model = model_class()
    model.load_state_dict(torch.load(filepath))
    model.eval()
    return model