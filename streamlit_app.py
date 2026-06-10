import os
import sys
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer
import streamlit as st

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))

from model import get_model


CHECKPOINT_PATH = "outputs/checkpoints/best_model.pt"
MAX_LEN         = 256

EXAMPLE_REVIEWS = [
    "This movie was absolutely fantastic. The acting was superb and the story kept me hooked from start to finish.",
    "Complete waste of time. The plot made no sense and the characters were painfully boring.",
    "One of the best films I have seen in years. Brilliant direction and an emotional storyline.",
    "Terrible CGI, weak script and awful pacing. I almost walked out halfway through.",
    "A masterpiece of modern cinema. Every scene was beautifully crafted and deeply moving.",
    "The worst movie I have ever seen. No story, no acting, just two hours of nothing.",
]


@st.cache_resource
def load_model():
    device     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer  = AutoTokenizer.from_pretrained("bert-base-uncased")
    model      = get_model()
    state_dict = torch.load(CHECKPOINT_PATH, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model, tokenizer, device


def predict(text, model, tokenizer, device):
    encoding = tokenizer(
        text,
        max_length=MAX_LEN,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )

    input_ids      = encoding["input_ids"].to(device)
    attention_mask = encoding["attention_mask"].to(device)

    with torch.no_grad():
        logits     = model(input_ids, attention_mask)
        probs      = F.softmax(logits, dim=1)
        confidence = probs.max().item()
        pred_class = torch.argmax(probs, dim=1).item()

    label           = "POSITIVE" if pred_class == 1 else "NEGATIVE"
    pos_confidence  = probs[0][1].item()
    neg_confidence  = probs[0][0].item()

    return {
        "label":          label,
        "confidence":     confidence,
        "positive_prob":  pos_confidence,
        "negative_prob":  neg_confidence,
    }

st.set_page_config(
    page_title="BERT Sentiment Analysis",
    page_icon="movie_camera",
    layout="wide",
)

with st.sidebar:
    st.title("Model Info")
    st.markdown("---")

    st.markdown("**Architecture**")
    st.code("bert-base-uncased", language=None)

    st.markdown("**Task**")
    st.write("Binary Sentiment Classification")

    st.markdown("**Dataset**")
    st.write("IMDB Movie Reviews")
    st.write("50,000 reviews (25k train / 25k test)")

    st.markdown("**Training**")
    st.write("3 epochs on Google Colab T4 GPU")
    st.write("AdamW optimizer, lr = 2e-5")
    st.write("Linear warmup scheduler")
    st.write("Mixed precision (fp16)")

    st.markdown("**Performance**")
    st.metric(label="Test Accuracy", value="93.4%")
    st.metric(label="F1 Score",      value="0.934")

    st.markdown("**Parameters**")
    st.write("109,483,778 total")
    st.write("109,483,778 trainable")

    st.markdown("---")
    st.caption("Fine-tuned BERT | PyTorch + HuggingFace")

st.title("BERT Sentiment Analysis")
st.write("Fine-tuned on IMDB movie reviews. Enter any review to classify it as positive or negative.")
st.markdown("---")

# Load model
with st.spinner("Loading model..."):
    try:
        model, tokenizer, device = load_model()
        st.success("Model loaded successfully")
    except FileNotFoundError:
        st.error(
            "Checkpoint not found at outputs/checkpoints/best_model.pt — "
            "please train the model first or place the checkpoint in the correct path."
        )
        st.stop()

st.subheader("Try an Example")
cols = st.columns(3)

selected_example = None

for i, example in enumerate(EXAMPLE_REVIEWS):
    col = cols[i % 3]
    with col:
        preview = example[:60] + "..." if len(example) > 60 else example
        if st.button(preview, key=f"example_{i}", use_container_width=True):
            selected_example = example

st.markdown("---")

st.subheader("Enter a Review")

default_text = selected_example if selected_example else ""

review_text = st.text_area(
    label="Movie review",
    value=default_text,
    placeholder="Type or paste a movie review here...",
    height=150,
    label_visibility="collapsed",
)

col_btn, col_clear, col_spacer = st.columns([1, 1, 6])

with col_btn:
    predict_clicked = st.button("Analyze Sentiment", type="primary", use_container_width=True)

with col_clear:
    if st.button("Clear", use_container_width=True):
        review_text = ""
        st.rerun()

if predict_clicked:
    if not review_text.strip():
        st.warning("Please enter a review first.")
    else:
        with st.spinner("Analyzing..."):
            result = predict(review_text, model, tokenizer, device)

        st.markdown("---")
        st.subheader("Result")

        left, right = st.columns([1, 2])

        with left:
            if result["label"] == "POSITIVE":
                st.success(f"POSITIVE")
            else:
                st.error(f"NEGATIVE")

            st.metric(
                label="Confidence",
                value=f"{result['confidence'] * 100:.1f}%",
            )

        with right:
            st.write("**Sentiment Probabilities**")

            st.write("Positive")
            st.progress(result["positive_prob"])
            st.caption(f"{result['positive_prob'] * 100:.1f}%")

            st.write("Negative")
            st.progress(result["negative_prob"])
            st.caption(f"{result['negative_prob'] * 100:.1f}%")

        st.markdown("---")

        with st.expander("Review submitted"):
            st.write(review_text)

        token_count = len(tokenizer.encode(review_text))
        st.caption(f"Token count : {token_count} / {MAX_LEN}")