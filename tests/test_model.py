import torch
import pytest
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from model import get_model, SentimentClassifier

class TestSentimentClassifier:

    def test_output_shape_is_batch_x_num_classes(self, model, dummy_input):
        """Forward pass should return (batch_size, 2) logits"""
        input_ids, attention_mask = dummy_input
        with torch.no_grad():
            logits = model(input_ids, attention_mask)
        assert logits.shape == (2, 2), f"Expected (2,2), got {logits.shape}"

    def test_output_is_tensor(self, model, dummy_input):
        input_ids, attention_mask = dummy_input
        with torch.no_grad():
            logits = model(input_ids, attention_mask)
        assert isinstance(logits, torch.Tensor)

    def test_output_is_not_nan(self, model, dummy_input):
        """NaN in logits means broken weights or gradient issues"""
        input_ids, attention_mask = dummy_input
        with torch.no_grad():
            logits = model(input_ids, attention_mask)
        assert not torch.isnan(logits).any(), "Logits contain NaN values"

    def test_output_is_not_inf(self, model, dummy_input):
        input_ids, attention_mask = dummy_input
        with torch.no_grad():
            logits = model(input_ids, attention_mask)
        assert not torch.isinf(logits).any(), "Logits contain Inf values"

    def test_different_inputs_give_different_outputs(self, model):
        """Model should not return identical logits for different inputs"""
        input_a = torch.randint(0, 1000, (1, 16))
        input_b = torch.randint(500, 2000, (1, 16))
        mask    = torch.ones(1, 16, dtype=torch.long)
        with torch.no_grad():
            out_a = model(input_a, mask)
            out_b = model(input_b, mask)
        assert not torch.allclose(out_a, out_b), \
            "Model returned identical outputs for different inputs"

    def test_batch_size_1_works(self, model):
        """Edge case: single sample batch"""
        input_ids      = torch.randint(0, 1000, (1, 16))
        attention_mask = torch.ones(1, 16, dtype=torch.long)
        with torch.no_grad():
            logits = model(input_ids, attention_mask)
        assert logits.shape == (1, 2)

    def test_larger_batch_size_works(self, model):
        """Edge case: larger batch"""
        input_ids      = torch.randint(0, 1000, (32, 16))
        attention_mask = torch.ones(32, 16, dtype=torch.long)
        with torch.no_grad():
            logits = model(input_ids, attention_mask)
        assert logits.shape == (32, 2)

    def test_eval_mode_disables_dropout(self, model, dummy_input):
        """In eval mode, two identical forward passes should give identical output"""
        input_ids, attention_mask = dummy_input
        model.eval()
        with torch.no_grad():
            out1 = model(input_ids, attention_mask)
            out2 = model(input_ids, attention_mask)
        assert torch.allclose(out1, out2), \
            "Eval mode produced different outputs for identical inputs — dropout not disabled"


class TestGetModel:

    def test_returns_sentiment_classifier_instance(self):
        m = get_model()
        assert isinstance(m, SentimentClassifier)

    def test_default_num_classes_is_2(self):
        m = get_model()
        assert m.linear.out_features == 2

    def test_custom_num_classes(self):
        m = get_model(num_classes=3)
        assert m.linear.out_features == 3

    def test_custom_dropout(self):
        m = get_model(dropout=0.5)
        assert m.dropout.p == 0.5

    def test_model_has_bert_layer(self):
        m = get_model()
        assert hasattr(m, "bert")

    def test_model_has_linear_layer(self):
        m = get_model()
        assert hasattr(m, "linear")

    def test_all_parameters_trainable_by_default(self):
        m = get_model()
        non_trainable = [n for n, p in m.named_parameters() if not p.requires_grad]
        assert len(non_trainable) == 0, \
            f"Found non-trainable parameters: {non_trainable}"