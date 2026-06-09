import os
import sys
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))

from model import get_model


CHECKPOINT_PATH = "outputs/checkpoints/best_model.pt"
MAX_LEN         = 256


def load_model(checkpoint_path):
    device     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer  = AutoTokenizer.from_pretrained("bert-base-uncased")
    model      = get_model()
    state_dict = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    print(f"Model loaded from : {checkpoint_path}")
    print(f"Device            : {device}\n")
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
        logits      = model(input_ids, attention_mask)
        probs       = F.softmax(logits, dim=1)
        confidence  = probs.max().item()
        pred_class  = torch.argmax(probs, dim=1).item()

    label = "POSITIVE" if pred_class == 1 else "NEGATIVE"

    return {"label": label, "confidence": confidence}


def main():
    print("Loading model...")
    model, tokenizer, device = load_model(CHECKPOINT_PATH)
    print("Model ready. Type a movie review to get sentiment prediction.")
    print("Type 'quit' to exit.\n")

    while True:
        text = input("Enter review: ").strip()

        if text.lower() == "quit":
            print("Exiting.")
            break

        if not text:
            print("Please enter some text.\n")
            continue

        result = predict(text, model, tokenizer, device)
        print(f"  {result['label']} (confidence: {result['confidence']*100:.1f}%)\n")


if __name__ == "__main__":
    main()