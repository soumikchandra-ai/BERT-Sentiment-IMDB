import torch
import torch.nn as nn
from transformers import AutoModel


class SentimentClassifier(nn.Module):

    def __init__(self, num_classes=2, dropout=0.3):
        super(SentimentClassifier, self).__init__()

        self.bert    = AutoModel.from_pretrained("bert-base-uncased")
        self.dropout = nn.Dropout(p=dropout)
        self.linear  = nn.Linear(self.bert.config.hidden_size, num_classes)

    def forward(self, input_ids, attention_mask):
        output     = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        
        # self.bert() returns a BaseModelOutputWithPoolingAndCrossAttentions object.
        # Grabbing last_hidden_state explicitly ensures we get the token embeddings tensor.
        cls_token  = output.last_hidden_state[:, 0, :]
        dropped    = self.dropout(cls_token)
        logits     = self.linear(dropped)
        return logits


def get_model(num_classes=2, dropout=0.3):
    model = SentimentClassifier(num_classes=num_classes, dropout=dropout)

    total     = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"Total parameters     : {total:,}")
    print(f"Trainable parameters : {trainable:,}")

    return model