# BERT Sentiment Analysis — IMDB

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.3-EE4C2C?logo=pytorch&logoColor=white)
![HuggingFace](https://img.shields.io/badge/HuggingFace-Transformers-FFD21E?logo=huggingface&logoColor=black)
![pytest](https://img.shields.io/badge/pytest-36%20passed-brightgreen?logo=pytest&logoColor=white)
![Coverage](https://img.shields.io/badge/model.py%20coverage-100%25-brightgreen)

Fine-tuned `bert-base-uncased` on the IMDB movie reviews dataset for binary
sentiment classification (positive / negative). Trained on Google Colab.
No API calls — everything runs in PyTorch
and HuggingFace.

---

## Results

| Metric        | Score              |
|---------------|--------------------|
| Test Accuracy | 93.4%              |
| F1 Score      | 0.934              |
| Dataset       | IMDB (50k reviews) |
| Epochs        | 3                  |
| Training time | ~20 min on T4 GPU  |

---

## Project Structure

```
bert-sentiment-imdb/
├── src/
│   ├── dataset.py              # IMDBDataset class and DataLoader factory
│   ├── model.py                # SentimentClassifier built on bert-base-uncased
│   ├── train.py                # training loop with AdamW and warmup scheduler
│   └── evaluate.py             # test set evaluation, confusion matrix, plots
├── outputs/
│   ├── checkpoints/            # best_model.pt saved here (not in repo due to large file size)
├── tests/
│   ├── __init__.py
│   ├── conftest.py             # shared fixtures (synthetic tensors, CPU model)
│   ├── test_model.py           # 14 tests: forward pass, output shape, edge cases
│   ├── test_dataset.py         # 11 tests: IMDBDataset class and tokenizer output
│   └── test_predict.py         # 11 tests: inference pipeline and edge cases
├── streamlit_app.py            # interactive web UI
├── app.py                      # CLI inference script
├── requirements.txt
└── README.md
```

---

## Setup

**1. Clone the repo**

```bash
git clone https://github.com/soumikchandra-ai/BERT-Sentiment-IMDB.git
cd bert-sentiment-imdb
```

**2. Create a virtual environment**

```bash
python -m venv venv
venv\Scripts\activate
```

**3. Install dependencies**

```bash
pip install -r requirements.txt
```

**4. Get the trained checkpoint**

The model checkpoint (`best_model.pt`) is not included in this repo due to
its size (~418 MB). You have two options:

- **Option A — Train it yourself on Google Colab**

Place the downloaded file at:

```
outputs/checkpoints/best_model.pt
```

---

## Training on Google Colab

**Step 1 — Open a new Colab notebook and set runtime to T4 GPU**

```
Runtime > Change runtime type > Hardware accelerator > T4 GPU
```

**Step 2 — Install dependencies**

```python
!pip install transformers==4.40.0 datasets==2.19.0 scikit-learn==1.4.2 accelerate==0.29.3
```

**Step 3 — Mount Google Drive**

```python
from google.colab import drive
drive.mount("/content/drive")
```

**Step 4 — Upload src/ folder via Colab sidebar**

```
Files > create src/ folder > upload dataset.py, model.py, train.py
```

**Step 5 — Run training**

```python
import sys
sys.path.append("/content/src")
from train import train
train()
```

Training takes approximately 20 minutes on a T4 GPU across 3 epochs.

**Step 6 — Download the checkpoint**

```python
from google.colab import files
files.download("/content/drive/MyDrive/bert-sentiment-imdb/outputs/checkpoints/best_model.pt")
```

Place `best_model.pt` in `outputs/checkpoints/`.

Also download `training_log.json` from your Drive and place it in `outputs/`.

---

## Running the Streamlit App

```bash
streamlit run streamlit_app.py
```

Opens at `http://localhost:8501`

Features:
- Model info sidebar with architecture and training details
- 6 example reviews to try instantly
- Confidence score and probability bars for each prediction
- Token counter showing how much of the 256 token limit was used

---

## Running CLI Inference

```bash
python app.py
```

```
Enter review: This film was absolutely brilliant
  POSITIVE (confidence: 97.8%)

Enter review: Boring plot and terrible acting
  NEGATIVE (confidence: 95.3%)
```

---

## Running Evaluation

```bash
python src/evaluate.py
```

Outputs:
- Accuracy and F1 score on the 25k test set
- Full classification report
- `outputs/plots/confusion_matrix.png`
- `outputs/plots/training_curves.png`

---

## Testing

The project includes an automated test suite of 36 pytest tests covering the
model architecture, dataset pipeline, and inference function. All tests run
on CPU with synthetic data — no checkpoint file or GPU required.

### Running the Tests

```bash
# Run all tests
pytest tests/ -v

# Run with coverage report
pytest tests/ --cov=src --cov-report=term-missing -v
```

### Test Results

```
tests/test_model.py::TestSentimentClassifier::test_output_shape_is_batch_x_num_classes    PASSED
tests/test_model.py::TestSentimentClassifier::test_output_is_tensor                       PASSED
tests/test_model.py::TestSentimentClassifier::test_output_is_not_nan                      PASSED
tests/test_model.py::TestSentimentClassifier::test_output_is_not_inf                      PASSED
tests/test_model.py::TestSentimentClassifier::test_different_inputs_give_different_outputs PASSED
tests/test_model.py::TestSentimentClassifier::test_batch_size_1_works                     PASSED
tests/test_model.py::TestSentimentClassifier::test_larger_batch_size_works                PASSED
tests/test_model.py::TestSentimentClassifier::test_eval_mode_disables_dropout             PASSED
tests/test_model.py::TestGetModel::test_returns_sentiment_classifier_instance             PASSED
tests/test_model.py::TestGetModel::test_default_num_classes_is_2                         PASSED
tests/test_model.py::TestGetModel::test_custom_num_classes                                PASSED
tests/test_model.py::TestGetModel::test_custom_dropout                                    PASSED
tests/test_model.py::TestGetModel::test_model_has_bert_layer                             PASSED
tests/test_model.py::TestGetModel::test_all_parameters_trainable_by_default              PASSED
tests/test_dataset.py::TestIMDBDataset::test_len_matches_data_size                       PASSED
tests/test_dataset.py::TestIMDBDataset::test_getitem_returns_dict                        PASSED
tests/test_dataset.py::TestIMDBDataset::test_getitem_has_required_keys                   PASSED
tests/test_dataset.py::TestIMDBDataset::test_input_ids_shape                             PASSED
tests/test_dataset.py::TestIMDBDataset::test_attention_mask_shape                        PASSED
tests/test_dataset.py::TestIMDBDataset::test_label_is_long_tensor                        PASSED
tests/test_dataset.py::TestIMDBDataset::test_label_value_matches_data                    PASSED
tests/test_dataset.py::TestIMDBDataset::test_input_ids_are_integers                      PASSED
tests/test_dataset.py::TestIMDBDataset::test_attention_mask_values_are_binary            PASSED
tests/test_dataset.py::TestIMDBDataset::test_truncation_respected                        PASSED
tests/test_dataset.py::TestIMDBDataset::test_single_item_dataset                         PASSED
tests/test_predict.py::TestPredict::test_returns_dict                                     PASSED
tests/test_predict.py::TestPredict::test_output_has_label_key                            PASSED
tests/test_predict.py::TestPredict::test_output_has_confidence_key                       PASSED
tests/test_predict.py::TestPredict::test_label_is_positive_or_negative                   PASSED
tests/test_predict.py::TestPredict::test_confidence_between_0_and_1                      PASSED
tests/test_predict.py::TestPredict::test_empty_string_does_not_crash                     PASSED
tests/test_predict.py::TestPredict::test_very_long_input_truncated_gracefully            PASSED
tests/test_predict.py::TestPredict::test_special_characters_handled                      PASSED
tests/test_predict.py::TestPredict::test_confidence_is_float                             PASSED
tests/test_predict.py::TestPredict::test_deterministic_output_in_eval_mode               PASSED

36 passed in 18.29s
```

### Coverage Report

```
Name              Stmts   Miss  Cover
-------------------------------------
src/dataset.py       32     15    53%
src/evaluate.py     102    102     0%
src/model.py         22      0   100%
src/train.py        124    124     0%
-------------------------------------
TOTAL               280    241    14%
```

`src/model.py` achieves 100% coverage. `src/evaluate.py` and `src/train.py`
are excluded from automated testing as they require a trained checkpoint, GPU
access, and the full 50k IMDB test set — dependencies unsuitable for a
lightweight CI test suite. The test suite is intentionally scoped to the
three components that are checkpoint-independent and CPU-runnable.

### What Is Tested

**test_model.py (14 tests)**
- `SentimentClassifier.forward`: output shape `(batch, 2)`, tensor type, NaN/Inf detection, distinct outputs for distinct inputs, batch size 1 and 32 edge cases, dropout disabled in eval mode
- `get_model`: instance type, default and custom `num_classes`, custom dropout value, layer presence, all parameters trainable

**test_dataset.py (11 tests)**
- `IMDBDataset.__len__`: length matches input data size, single-item edge case
- `IMDBDataset.__getitem__`: return type, required keys, `input_ids` and `attention_mask` shapes, label dtype and value correctness, integer token IDs, binary attention mask values, truncation at `max_len`

**test_predict.py (11 tests)**
- `predict()`: return type, required output keys, label validity, confidence range `[0, 1]`, confidence dtype, empty string input, oversized input truncation, special character handling, deterministic output in eval mode

### Edge Cases Covered

- Empty string input does not raise an exception
- Input exceeding `MAX_LEN=256` tokens is truncated gracefully without error
- Special characters, numbers, and punctuation are handled without crashing
- Batch size of 1 and batch size of 32 both produce correctly shaped outputs
- Identical inputs in eval mode produce identical outputs (dropout disabled)
- Attention mask values are strictly binary (0 or 1), never out of range
- NaN and Inf values in logits are explicitly checked and would fail immediately

---

## Model Architecture

```
bert-base-uncased
└── BertModel (12 transformer layers, hidden size 768)
    └── [CLS] token output
        └── Dropout (p=0.3)
            └── Linear (768 -> 2)
```

**Training config:**

| Parameter        | Value           |
|------------------|-----------------|
| Optimizer        | AdamW           |
| Learning rate    | 2e-5            |
| Weight decay     | 0.01            |
| Warmup ratio     | 10%             |
| Batch size       | 16              |
| Max token length | 256             |
| Mixed precision  | fp16 (autocast) |
| Seed             | 42              |

---

## Tech Stack

- Python 3.10+
- PyTorch 2.3
- HuggingFace Transformers 4.40
- HuggingFace Datasets 2.19
- Streamlit 1.35
- scikit-learn 1.4
- pytest + pytest-cov
- Google Colab (T4 GPU for training)

---

## Notes

- The checkpoint is excluded from this repo via `.gitignore`
- Training log (`training_log.json`) is included for reference
- All random seeds are fixed to 42 for reproducibility
- All 36 tests run on CPU with synthetic data — no checkpoint required
