import os
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import AutoTokenizer
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    confusion_matrix,
    classification_report,
)

import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from dataset import get_dataloaders
from model import get_model


def load_model(checkpoint_path, device):
    model      = get_model()
    state_dict = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    print(f"Model loaded from : {checkpoint_path}")
    return model


def get_predictions(model, loader, device):
    all_preds  = []
    all_labels = []

    with torch.no_grad():
        for batch in loader:
            input_ids      = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels         = batch["label"].to(device)

            logits = model(input_ids, attention_mask)
            preds  = torch.argmax(logits, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    return np.array(all_labels), np.array(all_preds)


def plot_confusion_matrix(labels, preds, save_path):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    cm = confusion_matrix(labels, preds)

    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Negative", "Positive"],
        yticklabels=["Negative", "Positive"],
    )
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix - BERT IMDB Sentiment")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Confusion matrix saved : {save_path}")


def plot_training_curves(log_path, save_path):
    if not os.path.exists(log_path):
        print(f"Training log not found at {log_path} - skipping curve plot")
        return

    with open(log_path, "r") as f:
        log = json.load(f)

    epochs     = [e["epoch"]      for e in log]
    train_loss = [e["train_loss"] for e in log]
    val_loss   = [e["val_loss"]   for e in log]
    train_acc  = [e["train_acc"]  for e in log]
    val_acc    = [e["val_acc"]    for e in log]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(epochs, train_loss, marker="o", label="Train Loss")
    axes[0].plot(epochs, val_loss,   marker="o", label="Val Loss")
    axes[0].set_title("Loss per Epoch")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()
    axes[0].grid(True)

    axes[1].plot(epochs, train_acc, marker="o", label="Train Accuracy")
    axes[1].plot(epochs, val_acc,   marker="o", label="Val Accuracy")
    axes[1].set_title("Accuracy per Epoch")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].legend()
    axes[1].grid(True)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Training curves saved  : {save_path}")


def evaluate():
    CHECKPOINT_PATH = "outputs/checkpoints/best_model.pt"
    LOG_PATH        = "outputs/training_log.json"
    CM_SAVE_PATH    = "outputs/plots/confusion_matrix.png"
    CURVE_SAVE_PATH = "outputs/plots/training_curves.png"
    MAX_LEN         = 256
    BATCH_SIZE      = 16

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device : {device}\n")

    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

    print("Loading test data...")
    _, _, test_loader = get_dataloaders(
        tokenizer,
        max_len=MAX_LEN,
        batch_size=BATCH_SIZE,
    )

    print("\nLoading model from checkpoint...")
    model = load_model(CHECKPOINT_PATH, device)

    print("\nRunning predictions on test set...")
    labels, preds = get_predictions(model, test_loader, device)

    accuracy = accuracy_score(labels, preds)
    f1       = f1_score(labels, preds, average="weighted")

    print("\n----- Evaluation Results -----")
    print(f"Test Accuracy : {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"F1 Score      : {f1:.4f}")
    print("\nClassification Report:")
    print(classification_report(labels, preds, target_names=["Negative", "Positive"]))

    plot_confusion_matrix(labels, preds, CM_SAVE_PATH)
    plot_training_curves(LOG_PATH, CURVE_SAVE_PATH)


if __name__ == "__main__":
    evaluate()