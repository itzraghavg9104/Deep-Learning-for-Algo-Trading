"""
Simplified LSTM model training for stock price prediction.

Uses vanilla PyTorch LSTM for reliable training.
"""
import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from sklearn.preprocessing import MinMaxScaler
import warnings
warnings.filterwarnings("ignore")


class StockDataset(Dataset):
    """Dataset for stock price sequences."""
    
    def __init__(self, sequences, targets):
        self.sequences = torch.FloatTensor(sequences)
        self.targets = torch.FloatTensor(targets)
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        return self.sequences[idx], self.targets[idx]


class StockLSTM(nn.Module):
    """LSTM model for stock price prediction."""
    
    def __init__(self, input_size=5, hidden_size=64, num_layers=2, output_size=1, dropout=0.2):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, output_size)
        )
    
    def forward(self, x):
        # x shape: (batch, seq_len, features)
        lstm_out, _ = self.lstm(x)
        # Take the last output
        last_out = lstm_out[:, -1, :]
        output = self.fc(last_out)
        return output


def prepare_sequences(df: pd.DataFrame, seq_length: int = 30, features: list = None) -> tuple:
    """
    Prepare sequences for LSTM training.
    
    Args:
        df: DataFrame with stock data
        seq_length: Sequence length for input
        features: List of feature columns
    
    Returns:
        Tuple of (sequences, targets, scaler)
    """
    if features is None:
        features = ['open', 'high', 'low', 'close', 'volume']
    
    all_sequences = []
    all_targets = []
    scalers = {}
    
    for symbol in df['symbol'].unique():
        symbol_data = df[df['symbol'] == symbol].sort_values('time_idx')
        
        # Scale data
        scaler = MinMaxScaler()
        scaled_data = scaler.fit_transform(symbol_data[features])
        scalers[symbol] = scaler
        
        # Create sequences
        for i in range(len(scaled_data) - seq_length):
            seq = scaled_data[i:i+seq_length]
            target = scaled_data[i+seq_length, 3]  # Close price (index 3)
            all_sequences.append(seq)
            all_targets.append(target)
    
    return np.array(all_sequences), np.array(all_targets), scalers


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int = 50,
    lr: float = 0.001,
    device: str = 'cpu'
) -> dict:
    """
    Train the LSTM model.
    
    Args:
        model: LSTM model
        train_loader: Training dataloader
        val_loader: Validation dataloader
        epochs: Number of epochs
        lr: Learning rate
        device: Device to use
    
    Returns:
        Training history
    """
    model = model.to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
    
    history = {'train_loss': [], 'val_loss': []}
    best_val_loss = float('inf')
    
    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0
        for sequences, targets in train_loader:
            sequences = sequences.to(device)
            targets = targets.to(device)
            
            optimizer.zero_grad()
            outputs = model(sequences)
            loss = criterion(outputs.squeeze(), targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        
        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for sequences, targets in val_loader:
                sequences = sequences.to(device)
                targets = targets.to(device)
                outputs = model(sequences)
                loss = criterion(outputs.squeeze(), targets)
                val_loss += loss.item()
        
        val_loss /= len(val_loader)
        scheduler.step(val_loss)
        
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), './models/lstm_best.pt')
        
        if (epoch + 1) % 5 == 0:
            print(f"Epoch [{epoch+1}/{epochs}] Train Loss: {train_loss:.6f} Val Loss: {val_loss:.6f}")
    
    return history


if __name__ == "__main__":
    print("=" * 50)
    print("LSTM Stock Price Prediction Training")
    print("=" * 50)
    
    # Config
    SEQ_LENGTH = 30
    BATCH_SIZE = 64
    EPOCHS = 30
    HIDDEN_SIZE = 64
    FEATURES = ['open', 'high', 'low', 'close', 'volume']
    
    # Device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Load data
    print("\nLoading data...")
    df = pd.read_csv('./data/training_data.csv')
    df['date'] = pd.to_datetime(df['date'])
    df = df.dropna()
    print(f"Loaded {len(df)} rows, {df['symbol'].nunique()} symbols")
    
    # Prepare sequences
    print("\nPreparing sequences...")
    sequences, targets, scalers = prepare_sequences(df, SEQ_LENGTH, FEATURES)
    print(f"Sequences shape: {sequences.shape}")
    print(f"Targets shape: {targets.shape}")
    
    # Split data
    split_idx = int(len(sequences) * 0.8)
    train_sequences, val_sequences = sequences[:split_idx], sequences[split_idx:]
    train_targets, val_targets = targets[:split_idx], targets[split_idx:]
    
    print(f"Training samples: {len(train_sequences)}")
    print(f"Validation samples: {len(val_sequences)}")
    
    # Create datasets and loaders
    train_dataset = StockDataset(train_sequences, train_targets)
    val_dataset = StockDataset(val_sequences, val_targets)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    # Create model
    os.makedirs('./models', exist_ok=True)
    model = StockLSTM(
        input_size=len(FEATURES),
        hidden_size=HIDDEN_SIZE,
        num_layers=2,
        output_size=1
    )
    print(f"\nModel parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Train
    print("\nStarting training...")
    history = train_model(
        model,
        train_loader,
        val_loader,
        epochs=EPOCHS,
        lr=0.001,
        device=device
    )
    
    # Save final model
    torch.save({
        'model_state_dict': model.state_dict(),
        'history': history,
        'config': {
            'seq_length': SEQ_LENGTH,
            'hidden_size': HIDDEN_SIZE,
            'features': FEATURES
        }
    }, './models/lstm_final.pt')
    
    print("\n" + "=" * 50)
    print("Training complete!")
    print(f"Best validation loss: {min(history['val_loss']):.6f}")
    print("Model saved to ./models/lstm_final.pt")
    print("=" * 50)
