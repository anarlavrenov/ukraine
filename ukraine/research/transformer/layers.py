import torch
from torch import nn
from torch.nn import functional as F
import math
from typing import Tuple, Optional


class Expert(nn.Module):
    """
    Represents a neural network expert model consisting of a sequence of linear
    layers with a ReLU activation function.

    This class initializes a simple neural network that processes input data
    through a series of transformations defined by the dimensions `d_model` and
    `dff`. The forward pass applies these transformations sequentially.

    :ivar net: A sequential container with layers including linear transformation,
        ReLU activation, and another linear transformation.
    :type net: nn.Sequential
    """
    def __init__(
            self,
            d_model: int,
            dff: int
    ):
        super(Expert, self).__init__()

        self.net = nn.Sequential(
            nn.Linear(d_model, dff),
            nn.ReLU(),
            nn.Linear(dff, d_model)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class NoisyTopkRouter(nn.Module):
    """
    Implements a Noisy Top-k Router for determining a sparse combination
    of expert distributions based on input data.

    The NoisyTopkRouter class employs a mechanism to add learned noise to the
    computed logits, enabling stochastic routing decisions. It selects the
    top-k logits, applies a mask to enforce sparsity in the routing
    distribution, and produces a softmax-normalized score for each expert.
    This class is useful for designing sparse mixture-of-experts architectures
    where only a subset of experts is activated for a given input.

    :ivar top_k: The number of top elements to be selected for sparse routing.
    :type top_k: int
    :ivar topkroute_linear: Fully connected layer that maps input to logits for routing.
    :type topkroute_linear: nn.Linear
    :ivar noise_linear: Fully connected layer that computes noise logits for stochastic routing.
    :type noise_linear: nn.Linear
    """
    def __init__(
            self,
            d_model: int,
            num_experts: int,
            top_k: int
    ):
        super(NoisyTopkRouter, self).__init__()

        self.top_k = top_k
        self.topkroute_linear = nn.Linear(d_model, num_experts)
        self.noise_linear = nn.Linear(d_model, num_experts)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        logits = self.topkroute_linear(x)
        noise_logits = self.noise_linear(x)

        noise = torch.randn_like(logits) * F.softplus(noise_logits)
        noisy_logits = logits + noise

        topk_logits, topk_indices = noisy_logits.topk(self.top_k, dim=-1)
        mask = torch.full_like(noisy_logits, fill_value=float("-inf"))
        sparse_logits = mask.scatter(-1, topk_indices, topk_logits)
        router_output = F.softmax(sparse_logits, dim=-1)

        return router_output, topk_indices


class SparseMoE(nn.Module):
    def __init__(
            self,
            d_model: int,
            dff: int,
            num_experts: int,
            top_k: int
    ):
        """
        Sparse Mixture of Experts(SparseMoE) initializes a sparse mixture of experts layer,
        utilizing a noisy top-k routing mechanism to distribute input among experts. Each
        expert operates independently, and only a specified top-k number of experts are
        selected to process any given input. This architecture enables efficient scaling in
        various machine learning tasks by maintaining a sparse activation pattern among
        the experts.

        :param d_model: Dimension of input data.
        :param dff: Dimension of feedforward network for each expert.
        :param num_experts: Total number of experts in the mixture.
        :param top_k: Number of experts to be selected for each input using top-k routing.

        """
        super(SparseMoE, self).__init__()

        self.top_k = top_k
        self.router = NoisyTopkRouter(d_model, num_experts, top_k)
        self.experts = nn.ModuleList([Expert(d_model, dff) for _ in range(num_experts)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Processes input through multiple experts, guided by a gating mechanism, and
        produces a weighted output based on gating scores.

        The method partitions the input based on gating results and routes portions
        of the data to respective experts. Each expert processes its designated
        portion, and the outputs are weighted and aggregated into the final result.

        :param x: The input tensor to the forward pass. Expected to be a multidimensional
            tensor where the last dimension corresponds to features.
        :type x: torch.Tensor
        :return: The aggregated output tensor after processing the input through
            experts and applying gating weights.
        :rtype: torch.Tensor
        """
        gating_output, indices = self.router(x)
        final_output = torch.zeros_like(x)

        flat_x = x.view(-1, x.size(-1))
        flat_gating_output = gating_output.view(-1, gating_output.size(-1))

        for i, expert in enumerate(self.experts):
            expert_mask = (indices == i).any(dim=-1)
            flat_mask = expert_mask.view(-1)

            if flat_mask.any():
                expert_input = flat_x[flat_mask]
                expert_output = expert(expert_input)
                gating_scores = flat_gating_output[flat_mask, i].unsqueeze(1)
                weighted_output = expert_output * gating_scores

                final_output[expert_mask] += weighted_output

        return final_output


class PositionalEncoding(torch.nn.Module):
    def __init__(
            self,
            d_model: int,
            max_len: int = 5000,
            scaled: bool = False
    ) -> None:
        """
            Initializes a positional encoding buffer for a PyTorch model.

            The class constructs and registers a buffer named `pe` that precomputes
            sine and cosine positional encodings, which can be used to provide a
            sense of order and positioning to transformer-based models. The encoding
            is constructed with alternating sine and cosine values based on the
            input dimensionality and the maximum sequence length, ensuring compatibility
            with self-attention mechanisms.

            :param d_model: Dimensionality of the feature/embedding space.
            :param max_len: Maximum sequence length for which positional encodings are precomputed.
                            Defaults to 1000.

            """
        super().__init__()

        self.scaled = scaled

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (
                    -math.log(10000.0) / d_model
            )
        )
        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

        if self.scaled:
            self.alpha = nn.Parameter(torch.ones(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Adds positional encoding to the input tensor.

        This method takes an input tensor and adds positional encoding to it, which
        is a technique commonly used in neural networks to encode the position of
        data in sequences. The positional encoding tensor is adjusted based on the
        input sequence length and added to the input tensor.

        :param x: Input tensor with positional embeddings.
        :type x: torch.Tensor.
        :return: torch.Tensor with positional encoding added to the input.
        :rtype: torch.Tensor.
        """
        if self.scaled:
            x = x + self.alpha * self.pe[:, :x.size(1)]
        else:
            x = x + self.pe[:, :x.size(1)]

        return x


class MultiheadAttention(nn.Module):
    def __init__(
            self,
            d_model: int,
            num_heads: int,
            dropout_rate: float,
            use_flash: bool = False
    ) -> None:
        """
        MultiheadAttention is a neural network layer designed to implement
        Scaled Dot-Product Multi-Head Attention as described in the Transformer
        architecture. It processes queries, keys, and values through multiple
        attention heads and combines their results. It internally computes the
        attention mechanism by linearly projecting input features to query, key,
        and value spaces, followed by scaled dot-product attention. This layer
        is useful for capturing dependencies across sequences.

        Attributes
        ----------
        d_model : int
            The dimensionality of the input embeddings to the attention mechanism.
        num_heads : int
            The number of attention heads to be utilized for multi-head attention.
        depth : int
            The projected dimensionality of each attention head.
        wq : nn.Linear
            The linear projection layer to compute query vectors.
        wk : nn.Linear
            The linear projection layer to compute key vectors.
        wv : nn.Linear
            The linear projection layer to compute value vectors.
        fc : nn.Linear
            The final fully connected layer applied to concatenated attention head
            outputs.
        dropout : nn.Dropout.
            Dropout layer applied post-attention for regularization.

        Parameters
        ----------
        :param d_model:
            The dimensionality of the input embedding space.
        :type d_model: int
        :param num_heads:
            Number of attention heads for the multi-head attention mechanism.
        :type num_heads: int
        :param dropout_rate:
            Dropout rate applied after attention computations for regularization.
        :type dropout_rate: float
        """
        super(MultiheadAttention, self).__init__()
        self.d_model = d_model
        self.num_heads = num_heads

        assert d_model % num_heads == 0
        self.depth = d_model // num_heads

        self.wq = nn.Linear(d_model, d_model, bias=False)
        self.wk = nn.Linear(d_model, d_model, bias=False)
        self.wv = nn.Linear(d_model, d_model, bias=False)

        self.fc = nn.Linear(d_model, d_model, bias=False)
        self.dropout = nn.Dropout(dropout_rate)
        self.use_flash = use_flash
        self.dropout_rate = dropout_rate

    def scaled_dot_product_attention(
            self,
            q: torch.Tensor,
            k: torch.Tensor,
            v: torch.Tensor,
            mask: Optional[torch.Tensor] = None,
            key_padding_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute the scaled dot-product attention mechanism. This is used
        within the attention mechanisms of deep learning models, particularly
        transformers. It computes attention scores by performing matrix
        multiplications between query, key, and value tensors, applies scaling,
        optional masking, and softmax normalization.

        :param q: Query tensor from the input sequence.
        :type q: torch.Tensor
        :param k: Key tensor from the input sequence.
        :type k: torch.Tensor
        :param v: Value tensor from the input sequence.
        :type v: torch.Tensor
        :param mask: Optional mask tensor to prevent attention over specific
            positions.
        :type mask: Optional[torch.Tensor]
        :param key_padding_mask: Optional padding mask to handle variable-length
            sequences, ensuring attention is not applied to padded indexes.
        :type key_padding_mask: Optional[torch.Tensor]
        :return: A tuple where the first tensor represents the weighted sum of
            values (output of the attention mechanism), and the second tensor
            represents the attention weights (scores).
        :rtype: Tuple[torch.Tensor, torch.Tensor]
        """
        matmul_qk = torch.matmul(q, torch.transpose(k, -2, -1))
        dk = k.size(-1)
        scaled_attention_logits = matmul_qk / math.sqrt(dk)

        if mask is not None:
            assert mask.dtype == torch.float
            assert mask.ndim == 2, "mask should be 2D tensor."
            assert mask.size(0) == scaled_attention_logits.size(-2)
            assert mask.size(1) == scaled_attention_logits.size(-1)
            mask = mask.unsqueeze(0).unsqueeze(0)

            scaled_attention_logits += mask

        if key_padding_mask is not None:
            assert key_padding_mask.dtype == torch.bool
            assert key_padding_mask.ndim == 2, "key_padding_mask should be 2D tensor."
            assert key_padding_mask.size(0) == scaled_attention_logits.size(0)
            assert key_padding_mask.size(1) == scaled_attention_logits.size(-1)

            key_padding_mask = key_padding_mask.unsqueeze(1).unsqueeze(2)
            scaled_attention_logits = scaled_attention_logits.masked_fill(
                key_padding_mask, float("-inf"))

        attention_weights = self.dropout(
            F.softmax(scaled_attention_logits, dim=-1))
        attention_output = torch.matmul(attention_weights, v)

        return attention_output, attention_weights

    def split_heads(
            self,
            x: torch.Tensor,
            batch_size: int
    ) -> torch.Tensor:
        """
        Splits the input tensor into multiple heads for multi-head attention mechanisms.
        This function reshapes the input tensor to separate the number of attention heads
        and the depth for each head, and then permutes the dimensions for appropriate
        computation within the attention mechanism.

        :param x: Input tensor of shape (batch_size, seq_length, num_heads * depth)
            to be reshaped and permuted.
        :param batch_size: Number of samples in the current batch.
        :return: Reshaped and permuted tensor of shape
            (batch_size, num_heads, seq_length, depth).
        """
        x = torch.reshape(x, [batch_size, -1, self.num_heads, self.depth])
        return torch.permute(x, [0, 2, 1, 3])

    def forward(self,
                q: torch.Tensor,
                k: torch.Tensor,
                v: torch.Tensor,
                mask: Optional[torch.Tensor] = None,
                key_padding_mask: Optional[torch.Tensor] = None
                ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Computes the forward pass of a multi-head attention layer. The method applies
        scaled dot-product attention followed by concatenation of attention heads and
        a linear transformation. It takes query, key, and value tensors as input along
        with optional masks for controlling which positions are attended to during
        computation.

        :param q: The query tensor of shape `(batch_size, seq_len, d_model)`, where
            `batch_size` is the size of the batch, `seq_len` is the sequence length,
            and `d_model` is the dimensionality of the model.
        :param k: The key tensor of shape `(batch_size, seq_len, d_model)`. It must
            match the query in terms of batch size and model dimensionality.
        :param v: The value tensor of shape `(batch_size, seq_len, d_model)`. It has
            the same shape as the key and query tensors.
        :param mask: An optional tensor of shape `(batch_size, num_heads, seq_len,
            seq_len)` used to mask certain positions in the sequence during attention
            computation. The default value is None.
        :param key_padding_mask: An optional tensor of shape `(batch_size, seq_len)`
            or `(batch_size, 1, seq_len)` used to mask padding positions. The
            default value is None.
        :return: A tuple where the first element is the output tensor of shape
            `(batch_size, seq_len, d_model)`, representing the processed values after
            attention and linear transformation, and the second element is the
            attention weights tensor of shape `(batch_size, num_heads, seq_len,
            seq_len)`.
        """
        batch_size = q.size(0)

        q = self.wq(q)
        k = self.wk(k)
        v = self.wv(v)

        q = self.split_heads(q, batch_size)
        k = self.split_heads(k, batch_size)
        v = self.split_heads(v, batch_size)

        if self.use_flash:

            B, _, Lq, _ = q.shape
            Lk = k.shape[-2]
            NEG_INF = torch.finfo(q.dtype).min
            attn_mask = None

            if mask is not None:
                assert mask.ndim == 2, "mask should be 2D tensor."
                attn_mask = mask.unsqueeze(0).unsqueeze(0) # (Lq, Lk) -> (1, 1, Lq, Lk)
                attn_mask = attn_mask.to(q.device, dtype=q.dtype)

            if key_padding_mask is not None:
                assert key_padding_mask.dtype == torch.bool, "key_padding_mask should be bool tensor."
                key_padding_mask = key_padding_mask.unsqueeze(1).unsqueeze(2) # (B, Lk) -> (B, 1, 1, Lk)
                key_padding_mask = key_padding_mask.to(q.device)
                kpm_f = torch.zeros(B, 1, 1, Lk, dtype=q.dtype, device=q.device).masked_fill(key_padding_mask, NEG_INF)
                attn_mask = kpm_f if attn_mask is None else (attn_mask + kpm_f)


            scaled_attention = nn.functional.scaled_dot_product_attention(
                q, k, v,
                attn_mask=attn_mask,
                dropout_p=self.dropout_rate if self.training else 0.0,
                is_causal=False
            )
            attention_weights = None

        else:

            scaled_attention, attention_weights = self.scaled_dot_product_attention(
                q, k, v, mask=mask, key_padding_mask=key_padding_mask
            )
        scaled_attention = torch.permute(scaled_attention, [0, 2, 1, 3])
        concat_attention = torch.reshape(
            scaled_attention, (batch_size, -1, self.d_model))
        output = self.fc(concat_attention)

        return output, attention_weights


class FeedForwardNetwork(nn.Module):
    def __init__(
            self,
            d_model,
            dff,
            dropout_rate: float = 0.1,
            activation: nn.Module = nn.GELU()
    ):
        super(FeedForwardNetwork, self).__init__()

        self.fc1 = nn.Linear(d_model, dff)
        self.fc2 = nn.Linear(dff, d_model)
        self.activation = activation
        self.dropout1 = nn.Dropout(dropout_rate)
        self.dropout2 = nn.Dropout(dropout_rate)

    def forward(self, x):
        x = self.fc2(self.dropout1(self.activation(self.fc1(x))))
        return self.dropout2(x)


class SiLUFeedForward(nn.Module):
    def __init__(
            self,
            d_model: int,
            dff: int,
            multiple_of: int = 4,
            ffn_dim_multiplier: Optional[float] = None
    ):
        super(SiLUFeedForward, self).__init__()

        dff = int(2 * dff / 3)

        if ffn_dim_multiplier is not None:
            dff = int(ffn_dim_multiplier * dff)
        dff = multiple_of * ((dff + multiple_of - 1) // multiple_of)

        self.w1 = nn.Linear(d_model, dff, bias=False)
        self.w2 = nn.Linear(dff, d_model, bias=False)
        self.w3 = nn.Linear(d_model, dff, bias=False)

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class DyT(nn.Module):
    """
    Applies a scaling, non-linear transformation, weighting, and biasing to
    input data. This process involves the following steps:

    1. The input tensor is first scaled by a learnable multiplier `alpha`,
       which is initialized to a specified value.
    2. The scaled tensor is passed through a hyperbolic tangent (tanh),
       which constrains its values to the range [-1, 1].
    3. The transformed tensor is then element-wise multiplied by a
       learnable weight vector.
    4. Finally, a learnable bias vector is added element-wise to the
       weighted tensor.

    This module enables controlling input magnitude and adapting flexible
    transformations of input data.

    :ivar alpha: A learnable scaling parameter initialized to the given value.
    :type alpha: torch.nn.Parameter
    :ivar weight: A learnable vector for element-wise weighting of the input.
    :type weight: torch.nn.Parameter
    :ivar bias: A learnable bias vector for shifting the output values.
    :type bias: torch.nn.Parameter
    """
    def __init__(
            self,
            num_features: int,
            alpha_init_value: float = 0.5
    ):
        """
        Сначала входные данные x масштабируются через коэффициент alpha_init_value.
        В дефолтном значении (0.5) все значения x просто уменьшатся в два раза.
        Потом данные проходят через тангенс,
        после чего все значения x будут находиться в диапазоне от -1 до 1.
        Потом x поэлементно взвешивается одномерным обучающимся
        вектором весов и добавляется одномерный
        обучающийся вектор смещения (bias).
        """
        super().__init__()

        self.alpha = nn.Parameter(torch.ones(1) * alpha_init_value)
        self.weight = nn.Parameter(torch.ones(num_features))
        self.bias = nn.Parameter(torch.zeros(num_features))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.tanh(self.alpha * x)
        return x * self.weight + self.bias
