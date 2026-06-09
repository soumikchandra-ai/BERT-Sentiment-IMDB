import os
import json
import random
import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from transformers import AutoTokenizer, get_linear_schedule_with_warmup
from torch.cuda.amp import GradScaler, autocast

import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from dataset import get_dataloaders
from model import get_model


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def train_one_epoch(model, loader, optimizer, scheduler, scaler, device, epoch):
    model.train()
    total_loss    = 0
    correct       = 0
    total_samples = 0
    criterion     = nn.CrossEntropyLoss()

    for step, batch in enumerate(loader):
        input_ids      = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels         = batch["label"].to(device)

        optimizer.zero_grad()

        with autocast():
            logits = model(input_ids, attention_mask)
            loss   = criterion(logits, labels)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        total_loss    += loss.item()
        preds          = torch.argmax(logits, dim=1)
        correct       += (preds == labels).sum().item()
        total_samples += labels.size(0)

        if (step + 1) % 100 == 0:
            avg_loss = total_loss / (step + 1)
            accuracy = correct / total_samples
            print(f"  Epoch {epoch} | Step {step+1}/{len(loader)} | Loss: {avg_loss:.4f} | Acc: {accuracy:.4f}")

    return total_loss / len(loader), correct / total_samples


def evaluate(model, loader, device):
    model.eval()
    total_loss    = 0
    correct       = 0
    total_samples = 0
    criterion     = nn.CrossEntropyLoss()

    with torch.no_grad():
        for batch in loader:
            input_ids      = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels         = batch["label"].to(device)

            with autocast():
                logits = model(input_ids, attention_mask)
                loss   = criterion(logits, labels)

            total_loss    += loss.item()
            preds          = torch.argmax(logits, dim=1)
            correct       += (preds == labels).sum().item()
            total_samples += labels.size(0)

    return total_loss / len(loader), correct / total_samples


def save_checkpoint(model, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(model.state_dict(), path)
    size_mb = os.path.getsize(path) / (1024 * 1024)
    print(f"Checkpoint saved : {path} ({size_mb:.1f} MB)")


def train():
    set_seed(42)

    EPOCHS         = 3
    BATCH_SIZE     = 16
    MAX_LEN        = 256
    LEARNING_RATE  = 2e-5
    WEIGHT_DECAY   = 0.01
    WARMUP_RATIO   = 0.1
    CHECKPOINT_DIR = "/content/drive/MyDrive/bert-sentiment-imdb/outputs/checkpoints"
    LOG_PATH       = "/content/drive/MyDrive/bert-sentiment-imdb/outputs/training_log.json"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device : {device}")
    print(f"GPU    : {torch.cuda.get_device_name(0)}\n")

    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

    print("Loading dataset...")
    train_loader, val_loader, _ = get_dataloaders(
        tokenizer,
        max_len=MAX_LEN,
        batch_size=BATCH_SIZE,
    )

    print("\nLoading model...")
    model = get_model()
    model = model.to(device)

    optimizer = AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    total_steps   = len(train_loader) * EPOCHS
    warmup_steps  = int(total_steps * WARMUP_RATIO)

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

    scaler = GradScaler()

    print(f"\nTotal steps   : {total_steps}")
    print(f"Warmup steps  : {warmup_steps}")
    print(f"Epochs        : {EPOCHS}")
    print(f"Batch size    : {BATCH_SIZE}")
    print(f"Learning rate : {LEARNING_RATE}\n")

    best_val_acc  = 0.0
    training_log  = []

    for epoch in range(1, EPOCHS + 1):
        print(f"--- Epoch {epoch}/{EPOCHS} ---")

        train_loss, train_acc = train_one_epoch(
            model, train_loader, optimizer, scheduler, scaler, device, epoch
        )

        val_loss, val_acc = evaluate(model, val_loader, device)

        print(f"\nEpoch {epoch} Summary:")
        print(f"  Train Loss : {train_loss:.4f}  |  Train Acc : {train_acc:.4f}")
        print(f"  Val Loss   : {val_loss:.4f}  |  Val Acc   : {val_acc:.4f}")

        epoch_log = {
            "epoch":      epoch,
            "train_loss": round(train_loss, 4),
            "train_acc":  round(train_acc,  4),
            }
        training_log.append(epoch_log)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            save_checkpoint(model, os.path.join(CHECKPOINT_DIR, "best_model.pt"))
            print(f"  New best val accuracy : {best_val_acc:.4f}")

        print()

    with open(LOG_PATH, "w") as f:
        json.dump(training_log, f, indent=2)
    print(f"Training log saved : {LOG_PATH}")

    print(f"\nTraining complete.")
    print(f"Best Val Accuracy : {best_val_acc:.4f}")
    print(f"Checkpoint saved to Google Drive.")


if __name__ == "__main__":
    train()