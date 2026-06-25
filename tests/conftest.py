import pytest
import torch
from transformers import AutoTokenizer


@pytest.fixture(scope="session")
def tokenizer():
    """Real tokenizer — loaded once for the whole test session"""
    return AutoTokenizer.from_pretrained("bert-base-uncased")


@pytest.fixture
def dummy_input():
    """Synthetic batch: 2 samples, seq_len=16 — no GPU, no checkpoint needed"""
    batch_size = 2
    seq_len    = 16
    input_ids      = torch.randint(0, 1000, (batch_size, seq_len))
    attention_mask = torch.ones(batch_size, seq_len, dtype=torch.long)
    return input_ids, attention_mask


@pytest.fixture
def dummy_batch():
    """Synthetic DataLoader-style batch dict"""
    batch_size = 4
    seq_len    = 16
    return {
        "input_ids":      torch.randint(0, 1000, (batch_size, seq_len)),
        "attention_mask": torch.ones(batch_size, seq_len, dtype=torch.long),
        "label":          torch.tensor([0, 1, 0, 1], dtype=torch.long),
    }


@pytest.fixture(scope="session")
def model():
    """CPU model — no checkpoint, no GPU required"""
    import sys, os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
    from model import get_model
    m = get_model()
    m.eval()
    return m