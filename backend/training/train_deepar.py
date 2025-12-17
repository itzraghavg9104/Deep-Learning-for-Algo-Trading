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
    
    # Recreate time index per symbol
    df["time_idx"] = df.groupby("symbol").cumcount()
    
    # Fill any NaN values
    df = df.fillna(method='ffill').fillna(method='bfill')
    
    # Ensure numeric columns
    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    df = df.dropna()
    
    return df


def create_datasets(
    df: pd.DataFrame,
    max_encoder_length: int = 30,
    max_prediction_length: int = 5,
    training_cutoff_ratio: float = 0.8
):
    """
    Create TimeSeriesDataSet for training and validation.
    """
    # Calculate training cutoff per group
    max_time_idx = df.groupby("symbol")["time_idx"].max().min()
    training_cutoff = int(max_time_idx * training_cutoff_ratio)
    
    print(f"Max time index: {max_time_idx}")
    print(f"Training cutoff: {training_cutoff}")
    
    # Filter training data
    train_df = df[df["time_idx"] <= training_cutoff].copy()
    
    # Create training dataset
    training = TimeSeriesDataSet(
        train_df,
        time_idx="time_idx",
        target="close",
        group_ids=["symbol"],
        min_encoder_length=max_encoder_length // 2,
        max_encoder_length=max_encoder_length,
        min_prediction_length=1,
        max_prediction_length=max_prediction_length,
        time_varying_known_reals=["time_idx"],
        time_varying_unknown_reals=["close", "volume"],
        target_normalizer="auto",
        add_relative_time_idx=True,
        add_encoder_length=True,
        allow_missing_timesteps=True,
    )
    
    # Create validation dataset
    validation = TimeSeriesDataSet.from_dataset(
        training,
        df,
        predict=True,
        stop_randomization=True,
    )
    
    return training, validation


def train_deepar(
    training: TimeSeriesDataSet,
    validation: TimeSeriesDataSet,
    hidden_size: int = 32,
    rnn_layers: int = 2,
    dropout: float = 0.1,
    learning_rate: float = 1e-3,
    max_epochs: int = 30,
    batch_size: int = 32,
    model_path: str = "./models"
) -> DeepAR:
    """
    Train DeepAR model.
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
        log_every_n_steps=10,
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
    print(f"Columns: {df.columns.tolist()}")
    
    print("\nCreating datasets...")
    training, validation = create_datasets(
        df,
        max_encoder_length=30,  # 30 days of history
        max_prediction_length=5,  # Predict 5 days ahead
    )
    print(f"Training samples: {len(training)}")
    print(f"Validation samples: {len(validation)}")
    
    print("\nTraining model...")
    model = train_deepar(
        training,
        validation,
        hidden_size=32,
        rnn_layers=2,
        max_epochs=30,
        batch_size=32,
        model_path="./models"
    )
    
    print("\n" + "=" * 50)
    print("Training complete!")
    print("=" * 50)
