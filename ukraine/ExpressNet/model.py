import torch
from torch import nn
from .layers import LinearLayer, BahdanauAttention


class ExpressNet(nn.Module):
    """

  Attributes:
    pre_lstm: LSTM layer that takes the initial embeddings as a tensor.
    post_lstm: LSTM layer that takes the concatenation of the context vector after BahdanauAttention
    and the initial embeddings as a tensor.
    fc: customized linear layer.
    bahdanau_attention: attention layer that takes the short-term memory of all LSTM cells
    and the short-term memory of only the last cell.

  """

    def __init__(
            self, d_model, vocab_size,
            classification_type: str, n_classes=None, recurrent_dropout=0.2
    ):
        super(ExpressNet, self).__init__()

        out_features = None
        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=d_model
        )

        self.pre_lstm = nn.LSTM(
            input_size=d_model,
            hidden_size=d_model,
            num_layers=3,
            bias=False,
            batch_first=True,
            bidirectional=False,
            dropout=recurrent_dropout
        )

        self.post_lstm = nn.LSTM(
            input_size=d_model,
            hidden_size=d_model,
            num_layers=2,
            bias=False,
            batch_first=True,
            bidirectional=True,
            dropout=recurrent_dropout
        )

        if classification_type == "binary":
            out_features = 1
        elif classification_type == "multiclass":
            assert n_classes is not None, ("for multiclass classification "
                                           "n_classes parameter must be specified.")
            out_features = n_classes

        self.fc = LinearLayer(
            in_features=d_model * 2,  # bidirectional lstm output
            out_features=out_features
        )

        self.bahdanau_attention = BahdanauAttention(d_model=d_model)

        self.dp = nn.Dropout(p=0.5)
        self.classification_type = classification_type

    def forward(self, x):
        """

    Args:
      x: torch.Tensor(batch_size, seq_len).

    Returns:
      torch.Tensor(batch_size, n_classes), where n_classes - is class logits.
    """
        output = None
        embedded = self.embedding(x)  # [batch_size, seq_len, d_model]
        embedded = self.dp(embedded)

        embeeded_lstm_out, (embeeded_ht, embeeded_ct) = self.pre_lstm(
            embedded)  # [batch_size, seq_len, d_model], [num_layers, batch_size, d_model]
        embeeded_ht = torch.mean(embeeded_ht, dim=0).\
            unsqueeze(0).permute(1, 0, 2)  # [batch_size, 1, d_model]

        context_vector = self.bahdanau_attention(
            embeeded_lstm_out, embeeded_ht)  # [batch_size, d_model]

        concated = torch.cat(
            [context_vector.unsqueeze(2),
             embedded.permute(0, 2, 1)],
            dim=-1)  # [batch_size, d_model, seq_len + 1]

        lstm_out, (_, __) = self.post_lstm(
            concated.permute(0, 2, 1))  # [batch_size, seq_len + 1, d_model * 2]
        lstm_out = torch.mean(lstm_out, dim=1)  # [batch_size, d_model * 2]

        if self.classification_type == "binary":
            output = nn.functional.sigmoid(self.fc(lstm_out))  # [batch_size, n_classes]
        elif self.classification_type == "multiclass":
            output = self.fc(lstm_out)

        return output