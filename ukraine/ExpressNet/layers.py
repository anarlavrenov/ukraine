import torch
from torch import nn


class LinearLayer(nn.Module):
    """
    A special linear layer used in Bahdanau Attention and the overall model.

    """

    def __init__(self, in_features, out_features):
        super(LinearLayer, self).__init__()

        self.weights = nn.Parameter(
            torch.Tensor(
                out_features, in_features
            )
        )  # [out_features, in_features]
        self.bias = nn.Parameter(
            torch.Tensor(
                out_features
            )
        )  # [out_features]

        self.xavier_uniform_()
        nn.init.uniform_(self.bias, -1e-4, 1e-4)

    def xavier_uniform_(self, gain=1):
        """
    Custom weight initialization function using the Xavier uniform method.
    Args:
      gain: Sigmoid activation multiplier
    """

        out_features, in_features = self.weights.size()

        a = gain * torch.sqrt(
            torch.tensor(6, dtype=torch.float32) / (in_features + out_features))
        with torch.no_grad():
            # noinspection PyArgumentList
            self.weights.uniform_(-a.item(), a.item())

    def forward(self, x):
        """
    According to the linear layer formula,
    the incoming data is multiplied by the transposed weights,
    followed by the addition of the bias.

    """
        if x.dim() == 2:
            x_times_w = torch.mm(x, self.weights.t())
            x_times_w_plus_b = x_times_w + self.bias

        elif x.dim() == 3:
            batch_size, seq_len, in_features = x.size()
            x = x.reshape(batch_size * seq_len, in_features)
            x_times_w = torch.mm(x, self.weights.t())
            x_times_w_plus_b = x_times_w + self.bias
            x_times_w_plus_b = x_times_w_plus_b.view(batch_size, seq_len, -1)

        else:
            raise ValueError(
                "The input tensor must have a dimension of 2D or 3D, "
                "but a {}D tensor was received.".format(x.dim())
            )

        return x_times_w_plus_b


class BahdanauAttention(nn.Module):
    def __init__(self, d_model):
        super(BahdanauAttention, self).__init__()

        self.d_model = d_model
        self.W1 = LinearLayer(
            in_features=d_model,
            out_features=d_model
        )
        self.W2 = LinearLayer(
            in_features=d_model,
            out_features=d_model
        )
        self.V = LinearLayer(
            in_features=d_model,
            out_features=1
        )

    def forward(self, query, values):
        attn_weights = nn.functional.softmax(
            self.V(
                torch.tanh(
                    self.W1(query) + self.W2(values)
                )
            ),
            dim=1
        )
        context_vector = attn_weights * values
        context_vector = torch.sum(context_vector, dim=1)

        return context_vector