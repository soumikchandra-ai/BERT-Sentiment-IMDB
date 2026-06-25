import torch
import pytest
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from dataset import IMDBDataset


# Minimal fake data — no HuggingFace download needed
FAKE_DATA = [
    {"text": "This movie was fantastic and I loved every moment of it.", "label": 1},
    {"text": "Terrible film. Complete waste of time.",                   "label": 0},
    {"text": "Brilliant acting and a gripping storyline throughout.",    "label": 1},
    {"text": "Boring, predictable, and painfully slow.",                 "label": 0},
]


class TestIMDBDataset:

    def test_len_matches_data_size(self, tokenizer):
        ds = IMDBDataset(FAKE_DATA, tokenizer, max_len=32)
        assert len(ds) == 4

    def test_getitem_returns_dict(self, tokenizer):
        ds   = IMDBDataset(FAKE_DATA, tokenizer, max_len=32)
        item = ds[0]
        assert isinstance(item, dict)

    def test_getitem_has_required_keys(self, tokenizer):
        ds   = IMDBDataset(FAKE_DATA, tokenizer, max_len=32)
        item = ds[0]
        assert "input_ids"      in item
        assert "attention_mask" in item
        assert "label"          in item

    def test_input_ids_shape(self, tokenizer):
        max_len = 32
        ds      = IMDBDataset(FAKE_DATA, tokenizer, max_len=max_len)
        item    = ds[0]
        assert item["input_ids"].shape == torch.Size([max_len])

    def test_attention_mask_shape(self, tokenizer):
        max_len = 32
        ds      = IMDBDataset(FAKE_DATA, tokenizer, max_len=max_len)
        item    = ds[0]
        assert item["attention_mask"].shape == torch.Size([max_len])

    def test_label_is_long_tensor(self, tokenizer):
        ds   = IMDBDataset(FAKE_DATA, tokenizer, max_len=32)
        item = ds[0]
        assert item["label"].dtype == torch.long

    def test_label_value_matches_data(self, tokenizer):
        ds = IMDBDataset(FAKE_DATA, tokenizer, max_len=32)
        assert ds[0]["label"].item() == 1   # positive
        assert ds[1]["label"].item() == 0   # negative

    def test_input_ids_are_integers(self, tokenizer):
        ds   = IMDBDataset(FAKE_DATA, tokenizer, max_len=32)
        item = ds[0]
        assert item["input_ids"].dtype in [torch.int32, torch.int64, torch.long]

    def test_attention_mask_values_are_binary(self, tokenizer):
        """Attention mask must only contain 0 or 1"""
        ds   = IMDBDataset(FAKE_DATA, tokenizer, max_len=32)
        item = ds[0]
        unique_vals = item["attention_mask"].unique().tolist()
        for v in unique_vals:
            assert v in [0, 1], f"Unexpected attention mask value: {v}"

    def test_truncation_respected(self, tokenizer):
        """Long text should be truncated to max_len, not exceed it"""
        long_text = "This is a great movie. " * 100   # very long
        data      = [{"text": long_text, "label": 1}]
        ds        = IMDBDataset(data, tokenizer, max_len=32)
        item      = ds[0]
        assert item["input_ids"].shape[0] == 32

    def test_single_item_dataset(self, tokenizer):
        data = [{"text": "Great film.", "label": 1}]
        ds   = IMDBDataset(data, tokenizer, max_len=32)
        assert len(ds) == 1
        item = ds[0]
        assert "input_ids" in item