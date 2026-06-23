import os
import math
import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


# ── Reproducibility ──────────────────────────────────────────────────────────

def set_seed(seed: int = 42):
    """Fix all random seeds for reproducible training."""
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ── Device ───────────────────────────────────────────────────────────────────

def get_device() -> torch.device:
    """Return GPU if available, else CPU."""
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():  # Apple Silicon
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")
    return device


# ── Checkpointing ────────────────────────────────────────────────────────────

def save_checkpoint(model, optimizer, epoch, loss, path="checkpoint.pth"):
    torch.save({
        "epoch": epoch,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "loss": loss,
    }, path)
    print(f"Checkpoint saved → {path}")


def load_checkpoint(model, optimizer, path="checkpoint.pth"):
    if not os.path.exists(path):
        print("No checkpoint found, starting fresh.")
        return 0, float("inf")
    ckpt = torch.load(path, map_location="cpu")
    model.load_state_dict(ckpt["model_state"])
    optimizer.load_state_dict(ckpt["optimizer_state"])
    print(f"Resumed from epoch {ckpt['epoch']} (loss: {ckpt['loss']:.5f})")
    return ckpt["epoch"], ckpt["loss"]


# ── Metrics ──────────────────────────────────────────────────────────────────

def evaluate_metrics(actual: np.ndarray, predicted: np.ndarray) -> dict:
    """Return RMSE, MAE, MAPE, and R² for inverse-transformed predictions."""
    rmse  = math.sqrt(mean_squared_error(actual, predicted))
    mae   = mean_absolute_error(actual, predicted)
    mape  = float(np.mean(np.abs((actual - predicted) / (actual + 1e-8))) * 100)
    r2    = r2_score(actual, predicted)
    metrics = {"RMSE": rmse, "MAE": mae, "MAPE %": mape, "R²": r2}
    for k, v in metrics.items():
        print(f"  {k:<8}: {v:.4f}")
    return metrics


# ── Early stopping ───────────────────────────────────────────────────────────

class EarlyStopping:
    """Stop training when validation loss stops improving."""

    def __init__(self, patience: int = 10, min_delta: float = 1e-5):
        self.patience   = patience
        self.min_delta  = min_delta
        self.best_loss  = float("inf")
        self.counter    = 0
        self.should_stop = False

    def step(self, val_loss: float) -> bool:
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter   = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                print(f"Early stopping triggered after {self.patience} epochs without improvement.")
                self.should_stop = True
        return self.should_stop


# ── Plotting ─────────────────────────────────────────────────────────────────

def plot_predictions(actual, predicted, ticker="STOCK", save_path="prediction_chart.png"):
    fig, axes = plt.subplots(2, 1, figsize=(12, 7), gridspec_kw={"height_ratios": [3, 1]})

    # Price chart
    axes[0].plot(actual,    label="Actual",    color="steelblue", linewidth=1.5)
    axes[0].plot(predicted, label="Predicted", color="coral",     linewidth=1.5, linestyle="--")
    axes[0].set_title(f"{ticker} — Predicted vs Actual Close Price", fontsize=13)
    axes[0].set_ylabel("Price (USD)")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    # Residuals
    residuals = np.array(actual).flatten() - np.array(predicted).flatten()
    axes[1].bar(range(len(residuals)), residuals, color=["#E24B4A" if r < 0 else "#1D9E75" for r in residuals], width=1.0)
    axes[1].axhline(0, color="gray", linewidth=0.8)
    axes[1].set_ylabel("Residual ($)")
    axes[1].set_xlabel("Trading days (test set)")
    axes[1].grid(alpha=0.2)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.show()
    print(f"Chart saved → {save_path}")


def plot_training_loss(train_losses, val_losses=None, save_path="loss_curve.png"):
    plt.figure(figsize=(8, 4))
    plt.plot(train_losses, label="Train loss", color="steelblue")
    if val_losses:
        plt.plot(val_losses, label="Val loss", color="coral", linestyle="--")
    plt.title("Training loss curve")
    plt.xlabel("Epoch")
    plt.ylabel("MSE loss")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.show()