"""
DeepAR model training script for stock price prediction.

Uses PyTorch Forecasting library for probabilistic forecasting.
"""
import os
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")

import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_forecasting import TimeSeriesDataSet, DeepAR
from pytorch_forecasting.data import NaNLabelEncoder
from pytorch_forecasting.metrics import NormalDistributionLoss


def load_training_data(data_path: str = "./data/training_data.csv") -> pd.DataFrame:
    """Load and prepare training data."""
    df = pd.read_csv(data_path)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["symbol", "date"]).reset_index(drop=True)
    
    # Recreate time index
    df["time_idx"] = df.groupby("symbol").cumcount()
    
    return df


def create_datasets(
    df: pd.DataFrame,
    max_encoder_length: int = 60,
    max_prediction_length: int = 5,
    training_cutoff_ratio: float = 0.8
):
    """
    Create TimeSeriesDataSet for training and validation.
    
    Args:
        df: Prepared DataFrame
        max_encoder_length: Number of historical days to use
        max_prediction_length: Number of days to predict
        training_cutoff_ratio: Train/validation split ratio
    
    Returns:
        Tuple of (training_dataset, validation_dataset)
    """
    # Calculate training cutoff
    training_cutoff = int(df["time_idx"].max() * training_cutoff_ratio)
    
    # Create training dataset
    training = TimeSeriesDataSet(
        df[df["time_idx"] <= training_cutoff],
        time_idx="time_idx",
        target="close",
        group_ids=["symbol"],
        max_encoder_length=max_encoder_length,
        max_prediction_length=max_prediction_length,
        time_varying_unknown_reals=["close", "open", "high", "low", "volume"],
        target_normalizer="auto",
        add_relative_time_idx=True,
        add_encoder_length=True,
        categorical_encoders={"symbol": NaNLabelEncoder(add_nan=True)},
    )
    
    # Create validation dataset using same parameters
    validation = TimeSeriesDataSet.from_dataset(
        training,
        df,
        min_prediction_idx=training_cutoff + 1,
    )
    
    return training, validation


def train_deepar(
    training: TimeSeriesDataSet,
    validation: TimeSeriesDataSet,
    hidden_size: int = 64,
    rnn_layers: int = 2,
    dropout: float = 0.1,
    learning_rate: float = 1e-3,
    max_epochs: int = 50,
    batch_size: int = 64,
    model_path: str = "./models"
) -> DeepAR:
    """
    Train DeepAR model.
    
    Args:
        training: Training dataset
        validation: Validation dataset
        hidden_size: LSTM hidden size
        rnn_layers: Number of RNN layers
        dropout: Dropout rate
        learning_rate: Learning rate
        max_epochs: Maximum training epochs
        batch_size: Batch size
        model_path: Path to save model
    
    Returns:
        Trained DeepAR model
    """
    os.makedirs(model_path, exist_ok=True)
    
    # Create dataloaders
    train_dataloader = training.to_dataloader(
        train=True,
        batch_size=batch_size,
        num_workers=0
    )
    val_dataloader = validation.to_dataloader(
        train=False,
        batch_size=batch_size,
        num_workers=0
    )
    
    # Create model
    model = DeepAR.from_dataset(
        training,
        hidden_size=hidden_size,
        rnn_layers=rnn_layers,
        dropout=dropout,
        learning_rate=learning_rate,
        loss=NormalDistributionLoss(),
        log_interval=10,
        reduce_on_plateau_patience=3,
    )
    
    print(f"Model parameters: {model.size()/1e3:.1f}k")
    
    # Setup callbacks
    early_stop = EarlyStopping(
        monitor="val_loss",
        patience=5,
        mode="min"
    )
    
    checkpoint = ModelCheckpoint(
        dirpath=model_path,
        filename="deepar-{epoch:02d}-{val_loss:.4f}",
        monitor="val_loss",
        mode="min",
        save_top_k=1
    )
    
    # Create trainer
    trainer = pl.Trainer(
        max_epochs=max_epochs,
        accelerator="auto",
        gradient_clip_val=0.1,
        callbacks=[early_stop, checkpoint],
        enable_progress_bar=True,
        logger=True,
    )
    
    # Train
    print("Starting training...")
    trainer.fit(
        model,
        train_dataloaders=train_dataloader,
        val_dataloaders=val_dataloader
    )
    
    # Load best checkpoint
    best_model_path = checkpoint.best_model_path
    print(f"Best model saved to: {best_model_path}")
    
    # Save for inference
    torch.save(model.state_dict(), os.path.join(model_path, "deepar_final.pt"))
    
    return model


def predict(
    model: DeepAR,
    dataset: TimeSeriesDataSet,
    return_x: bool = True
):
    """
    Make predictions with trained model.
    
    Args:
        model: Trained DeepAR model
        dataset: Dataset to predict on
        return_x: Whether to return input data
    
    Returns:
        Predictions DataFrame
    """
    predictions = model.predict(
        dataset.to_dataloader(train=False, batch_size=64),
        return_x=return_x,
        mode="prediction"
    )
    
    return predictions


if __name__ == "__main__":
    print("=" * 50)
    print("DeepAR Stock Price Forecasting Training")
    print("=" * 50)
    
    # Load data
    data_path = "./data/training_data.csv"
    
    if not os.path.exists(data_path):
        print(f"Data file not found: {data_path}")
        print("Run download_data.py first to download training data.")
        exit(1)
    
    print("Loading data...")
    df = load_training_data(data_path)
    print(f"Loaded {len(df)} rows, {df['symbol'].nunique()} symbols")
    
    print("\nCreating datasets...")
    training, validation = create_datasets(
        df,
        max_encoder_length=60,  # 60 days of history
        max_prediction_length=5,  # Predict 5 days ahead
    )
    print(f"Training samples: {len(training)}")
    print(f"Validation samples: {len(validation)}")
    
    print("\nTraining model...")
    model = train_deepar(
        training,
        validation,
        hidden_size=64,
        rnn_layers=2,
        max_epochs=50,
        batch_size=64,
        model_path="./models"
    )
    
    print("\nTraining complete!")
