import torch
import torch.nn as nn


# =========================
# ATTENTION LAYER
# =========================

class Attention(nn.Module):
    def __init__(self, hidden_size):
        super(Attention, self).__init__()

        self.attention = nn.Linear(hidden_size * 2, hidden_size * 2)
        self.context = nn.Linear(hidden_size * 2, 1, bias=False)

    def forward(self, x):
        """
        x shape:
        (batch_size, seq_len, hidden_size*2)
        """

        score = torch.tanh(self.attention(x))

        attention_weights = torch.softmax(
            self.context(score),
            dim=1
        )

        context_vector = torch.sum(attention_weights * x, dim=1)

        return context_vector, attention_weights


# =========================
# SARCASM MODEL
# =========================

class SarcasmModel(nn.Module):

    def __init__(
        self,
        vocab_size,
        embedding_dim=300,
        hidden_size=128,
        dropout=0.3
    ):

        super(SarcasmModel, self).__init__()

        # Embedding Layer
        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embedding_dim,
            padding_idx=0
        )

        # BiGRU
        self.bigru = nn.GRU(
            embedding_dim,
            hidden_size,
            batch_first=True,
            bidirectional=True
        )

        # Attention
        self.attention = Attention(hidden_size)

        # Dropout
        self.dropout = nn.Dropout(dropout)

        # Output layer
        self.fc = nn.Linear(hidden_size * 2, 1)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):

        # x shape: (batch_size, seq_len)

        embedded = self.embedding(x)

        # (batch_size, seq_len, embed_dim)

        gru_output, _ = self.bigru(embedded)

        # (batch_size, seq_len, hidden*2)

        context_vector, attention_weights = self.attention(gru_output)

        x = self.dropout(context_vector)

        logits = self.fc(x)

        output = self.sigmoid(logits)

        return output