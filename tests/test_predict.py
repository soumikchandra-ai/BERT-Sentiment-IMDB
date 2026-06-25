import torch
import pytest
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ))

from app import predict


class TestPredict:
    """Tests for the predict() function in app.py
       Uses a CPU model — no checkpoint file needed."""

    @pytest.fixture
    def loaded_model(self, model, tokenizer):
        device = torch.device("cpu")
        model.to(device)
        return model, tokenizer, device

    def test_returns_dict(self, loaded_model):
        m, tok, dev = loaded_model
        result = predict("Great movie!", m, tok, dev)
        assert isinstance(result, dict)

    def test_output_has_label_key(self, loaded_model):
        m, tok, dev = loaded_model
        result = predict("Great movie!", m, tok, dev)
        assert "label" in result

    def test_output_has_confidence_key(self, loaded_model):
        m, tok, dev = loaded_model
        result = predict("Great movie!", m, tok, dev)
        assert "confidence" in result

    def test_label_is_positive_or_negative(self, loaded_model):
        m, tok, dev = loaded_model
        result = predict("Amazing film, loved it!", m, tok, dev)
        assert result["label"] in ["POSITIVE", "NEGATIVE"]

    def test_confidence_between_0_and_1(self, loaded_model):
        m, tok, dev = loaded_model
        result = predict("Terrible movie.", m, tok, dev)
        assert 0.0 <= result["confidence"] <= 1.0

    def test_empty_string_does_not_crash(self, loaded_model):
        """Edge case: empty input should not raise an exception"""
        m, tok, dev = loaded_model
        try:
            result = predict("", m, tok, dev)
            assert "label" in result
        except Exception as e:
            pytest.fail(f"predict() raised exception on empty string: {e}")

    def test_very_long_input_truncated_gracefully(self, loaded_model):
        """Input longer than MAX_LEN should be truncated, not crash"""
        m, tok, dev = loaded_model
        long_text   = "This movie was great. " * 300
        result      = predict(long_text, m, tok, dev)
        assert result["label"] in ["POSITIVE", "NEGATIVE"]

    def test_special_characters_handled(self, loaded_model):
        """Input with special chars, numbers, punctuation should not crash"""
        m, tok, dev = loaded_model
        result      = predict("!!! @#$% 123 --- great???", m, tok, dev)
        assert "label" in result

    def test_confidence_is_float(self, loaded_model):
        m, tok, dev = loaded_model
        result      = predict("Good film.", m, tok, dev)
        assert isinstance(result["confidence"], float)

    def test_deterministic_output_in_eval_mode(self, loaded_model):
        """Same input twice should give identical predictions"""
        m, tok, dev = loaded_model
        text        = "One of the best films ever made."
        result1     = predict(text, m, tok, dev)
        result2     = predict(text, m, tok, dev)
        assert result1["label"]      == result2["label"]
        assert result1["confidence"] == result2["confidence"]