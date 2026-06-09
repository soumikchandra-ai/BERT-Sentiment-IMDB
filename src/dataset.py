import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
from datasets import load_dataset


class IMDBDataset(Dataset):

    def __init__(self, data, tokenizer, max_len=256):
        self.data      = data
        self.tokenizer = tokenizer
        self.max_len   = max_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        text  = self.data[idx]["text"]
        label = self.data[idx]["label"]

        encoding = self.tokenizer(
            text,
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        return {
            "input_ids":      encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "label":          torch.tensor(label, dtype=torch.long),
        }


def get_dataloaders(tokenizer, max_len=256, batch_size=16, val_split=0.1):

    raw_dataset = load_dataset("stanfordnlp/imdb")   # fixed

    split      = raw_dataset["train"].train_test_split(test_size=val_split, seed=42)
    train_data = split["train"]
    val_data   = split["test"]
    test_data  = raw_dataset["test"]

    train_dataset = IMDBDataset(train_data, tokenizer, max_len)
    val_dataset   = IMDBDataset(val_data,   tokenizer, max_len)
    test_dataset  = IMDBDataset(test_data,  tokenizer, max_len)

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,  num_workers=2, pin_memory=True
    )
    val_loader   = DataLoader(
        val_dataset,   batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True
    )
    test_loader  = DataLoader(
        test_dataset,  batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True
    )

    print(f"Train : {len(train_dataset)} samples")
    print(f"Val   : {len(val_dataset)} samples")
    print(f"Test  : {len(test_dataset)} samples")

    return train_loader, val_loader, test_loader