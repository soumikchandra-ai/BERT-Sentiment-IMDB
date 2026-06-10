# BERT Sentiment Analysis — IMDB

Fine-tuned `bert-base-uncased` on the IMDB movie reviews dataset for binary
sentiment classification (positive / negative). Trained on Google Colab,
deployed locally via Streamlit. No API calls — everything runs in PyTorch
and HuggingFace.

---

## Results

| Metric        | Score  |
|---------------|--------|
| Test Accuracy | 93.4%  |
| F1 Score      | 0.934  |
| Dataset       | IMDB (50k reviews) |
| Epochs        | 3      |
| Training time | ~65 min on T4 GPU  |

---

## Project Structure

```
bert-sentiment-imdb/
├── notebooks/
│   └── exploration.ipynb       # data exploration and token length analysis
├── src/
│   ├── dataset.py              # IMDBDataset class and DataLoader factory
│   ├── model.py                # SentimentClassifier built on bert-base-uncased
│   ├── train.py                # training loop with AdamW and warmup scheduler
│   └── evaluate.py             # test set evaluation, confusion matrix, plots
├── outputs/
│   ├── checkpoints/            # best_model.pt saved here (not in repo)
│   └── plots/                  # confusion matrix and training curves
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

- **Option A — Train it yourself on Google Colab** (see Training section below)
- **Option B — Download from Google Drive** — [best_model.pt](#) *(add your Drive link here)*

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

Training takes approximately 65 minutes on a T4 GPU across 3 epochs.

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

## Model Architecture

```
bert-base-uncased
└── BertModel (12 transformer layers, hidden size 768)
    └── [CLS] token output
        └── Dropout (p=0.3)
            └── Linear (768 -> 2)
```

**Training config:**

| Parameter         | Value          |
|-------------------|----------------|
| Optimizer         | AdamW          |
| Learning rate     | 2e-5           |
| Weight decay      | 0.01           |
| Warmup ratio      | 10%            |
| Batch size        | 16             |
| Max token length  | 256            |
| Mixed precision   | fp16 (autocast)|
| Seed              | 42             |

---

## Tech Stack

- Python 3.10+
- PyTorch 2.3
- HuggingFace Transformers 4.40
- HuggingFace Datasets 2.19
- Streamlit 1.35
- scikit-learn 1.4
- Google Colab (T4 GPU for training)

---

## Notes

- The checkpoint is excluded from this repo via `.gitignore`
- Training log (`training_log.json`) is included for reference
- All random seeds are fixed to 42 for reproducibility