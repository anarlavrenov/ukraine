import torch
from torch import nn
from torch.nn import functional as F
import math
from typing import Tuple, Optional, Dict


class PositionalEncoding(torch.nn.Module):
    def __init__(
            self,
            d_model: int,
            max_len: int = 1000
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

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (
                    -math.log(10000.0) / d_model
            )
        )
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

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
        x = x + self.pe[:x.size(1)].transpose(0, 1)

        return x


class MultiheadAttention(nn.Module):
    def __init__(
            self,
            d_model: int,
            num_heads: int,
            dropout_rate: float
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

        self.wq = nn.Linear(d_model, d_model)
        self.wk = nn.Linear(d_model, d_model)
        self.wv = nn.Linear(d_model, d_model)

        self.fc = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout_rate)

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

        scaled_attention, attention_weights = self.scaled_dot_product_attention(
            q, k, v, mask=mask, key_padding_mask=key_padding_mask
        )
        scaled_attention = torch.permute(scaled_attention, [0, 2, 1, 3])
        concat_attention = torch.reshape(
            scaled_attention, (batch_size, -1, self.d_model))
        output = self.fc(concat_attention)

        return output, attention_weights


class EncoderLayer(nn.Module):
    def __init__(
            self,
            d_model: int,
            num_heads: int,
            dff: int,
            dropout_rate: float,
            activation: nn.Module = nn.ReLU(),
    ) -> None:
        """
        Represents a single encoder layer used in a transformer model. This layer is composed
        of a multi-head attention sub-layer and a feedforward network. Residual connections
        along with layer normalization are applied after each sub-layer. Dropout is used to
        help regularize the network during training.

        :param d_model: The dimensionality of the input and output layers.
        :type d_model: int
        :param num_heads: The number of attention heads used in the multi-head
            attention mechanism.
        :type num_heads: int
        :param dff: The dimensionality of the feedforward network's hidden layer.
        :type dff: int
        :param dropout_rate: The probability of an element to be set to zero
            during dropout regularization.
        :type dropout_rate: float
        :param activation: The activation function module to apply in the feedforward
            network. Default is ReLU.
        :type activation: nn.Module
        """
        super(EncoderLayer, self).__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.dff = dff
        self.mha = MultiheadAttention(d_model, num_heads, dropout_rate)

        self.fc1 = nn.Linear(d_model, dff)
        self.fc2 = nn.Linear(dff, d_model)
        self.activation = activation
        self.ffn_dropout1 = nn.Dropout(dropout_rate)
        self.ffn_dropout2 = nn.Dropout(dropout_rate)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout_rate)

    def ff_block(self, x: torch.Tensor) -> torch.Tensor:
        """
        Processes the input tensor through feed-forward neural network layers including
        activation and dropout mechanisms.

        :param x: The input tensor to be processed.
        :type x: torch.Tensor
        :return: The output tensor after processing through the feed-forward network.
        :rtype: torch.Tensor
        """
        x = self.fc2(self.ffn_dropout1(self.activation(self.fc1(x))))
        return self.ffn_dropout2(x)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None,
                src_key_padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Computes the forward pass of a Transformer encoder layer, where
        multi-head attention is applied followed by normalization, dropout,
        and a feed-forward block. The method produces an output tensor,
        which corresponds to the encoded representation of the input.

        :param x: Input tensor to the Transformer layer.
        :param mask: Optional attention mask tensor applied to the source.
            Default is None.
        :param src_key_padding_mask: Optional padding mask tensor used to mask
            out padding positions. Default is None.
        :return: Encoded output tensor resulting from applying the Transformer
            encoder layer to the input tensor.
        """
        attention_output, attention_weights = self.mha(
            x, x, x, mask, src_key_padding_mask)
        out1 = self.norm1(x + self.dropout(attention_output))
        out2 = self.norm2(out1 + self.ff_block(out1))

        return out2


class DecoderLayer(nn.Module):
    def __init__(
            self,
            d_model: int,
            num_heads: int,
            dff: int,
            dropout_rate: float,
            activation: nn.Module = nn.ReLU()):
        """
        Initializes an instance of the DecoderLayer class utilized in Transformer-based
        architectures. The class encapsulates the self-attention, cross-attention
        mechanisms along with a feedforward neural network layer. Additionally, it
        incorporates dropout and normalization layers to assist in regularization and
        stabilize training dynamics.

        :param d_model: The dimensionality of the input embeddings. Represents the width
            of attention layers and intermediary feedforward layers.
        :param num_heads: Number of attention heads in the multihead attention layers.
        :param dff: Dimensionality of the hidden layer in the feedforward neural network.
        :param dropout_rate: Dropout probability applied across various attentional
            layers and intermediate feedforward connections.
        :param activation: Activation function used in the feedforward neural network,
            defaulted to nn.ReLU().
        """
        super(DecoderLayer, self).__init__()
        self.self_attention = MultiheadAttention(d_model, num_heads, dropout_rate)
        self.cross_attention = MultiheadAttention(d_model, num_heads, dropout_rate)

        self.fc1 = nn.Linear(d_model, dff)
        self.fc2 = nn.Linear(dff, d_model)
        self.activation = activation
        self.ffn_dropout1 = nn.Dropout(dropout_rate)
        self.ffn_dropout2 = nn.Dropout(dropout_rate)

        self.dropout1 = nn.Dropout(dropout_rate)
        self.dropout2 = nn.Dropout(dropout_rate)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)

    def ff_block(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies a feedforward block to the input tensor. This function is designed to combine linear
        transformations, an activation function, and dropout layers to process the input tensor and
        produce the output tensor. The computation comprises a sequence of fully connected layers,
        dropout operations, and activation applied to the given input tensor.

        :param x: The input tensor to be processed by the feedforward block - the output
            of a multi-head attention mechanism.
        :type x: torch.Tensor

        :return: The processed tensor after fully connected layers, activation, and dropout
            operations, representing the output of the feedforward block.
        :rtype: torch.Tensor
        """
        x = self.fc2(self.ffn_dropout1(self.activation(self.fc1(x))))
        return self.ffn_dropout2(x)

    def forward(
            self,
            x: torch.Tensor,
            memory: torch.Tensor,
            tgt_mask: Optional[torch.Tensor] = None,
            memory_mask: Optional[torch.Tensor] = None,
            tgt_key_padding_mask: Optional[torch.Tensor] = None,
            memory_key_padding_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Performs a forward pass through the Transformer decoder layer. The decoder layer
        includes a self-attention mechanism, a cross-attention mechanism, and a feedforward
        block. It normalizes outputs of each stage and applies dropout for regularization.

        :param x: Input to the decoder layer, representing target sequences,
            with shape `(batch_size, target_seq_len, embed_dim)` in a
            Transformer-based architecture.
        :param memory: Memory input from the encoder, representing source sequences,
            with shape `(batch_size, target_seq_len, embed_dim)`.
        :param tgt_mask: Optional mask used in the self-attention mechanism to prevent
            attention to certain positions in the target sequence. Shape
            `(target_seq_len, target_seq_len)`.
        :param memory_mask: Optional mask used in the cross-attention mechanism to
            prevent attention to certain positions in the source sequence. Shape
            `(target_seq_len, source_seq_len)`.
        :param tgt_key_padding_mask: Optional mask indicating which positions in the
            target sequence should be ignored during self-attention calculation. Shape
            `(batch_size, target_seq_len)`.
        :param memory_key_padding_mask: Optional mask indicating which positions in the
            source sequence should be ignored during cross-attention calculation. Shape
            `(batch_size, source_seq_len)`.
        :return: Returns a tuple containing the following:

            - out3: The output tensor resulting from the decoder layer, after applying
              self-attention, cross-attention, and the feedforward block, with shape
              `(batch_size, target_seq_len, embed_dim)`.

            - self_attention_weights: The attention weights produced by the self-attention
              mechanism, with shape `(batch_size, num_heads, target_seq_len, target_seq_len)`.

            - cross_attention_weights: The attention weights produced by the cross-attention
              mechanism, with shape `(batch_size, num_heads, target_seq_len, source_seq_len)`.
        """
        self_attention_output, self_attention_weights = self.self_attention(
            x, x, x, mask=tgt_mask, key_padding_mask=tgt_key_padding_mask)
        out1 = self.norm1(x + self.dropout1(self_attention_output))

        cross_attention_output, cross_attention_weights = self.cross_attention(
            out1, memory, memory, mask=memory_mask, key_padding_mask=memory_key_padding_mask)
        out2 = self.norm2(out1 + self.dropout2(cross_attention_output))

        out3 = self.norm3(out2 + self.ff_block(out2))

        return out3, self_attention_weights, cross_attention_weights


class Encoder(nn.Module):
    def __init__(
            self,
            num_encoder_layers: int,
            d_model: int,
            num_heads: int,
            dff: int,
            input_vocab_size: int,
            dropout_rate: float,
            activation: nn.Module = nn.ReLU()
    ) -> None:
        """
        Initializes the Encoder with specified parameters and creates required
        attributes such as positional encoding, embedding, and encoder layers
        using the provided configurations. This class sets up the building blocks
        for a Transformer Encoder module.

        :param num_encoder_layers: The number of encoder layers in the model.
        :param d_model: The dimension of the embedding space and internal model representation.
        :param num_heads: The number of attention heads in the multi-head attention mechanism.
        :param dff: The dimensionality of the feed-forward network in each encoder layer.
        :param input_vocab_size: The size of the input vocabulary for the embedding layer.
        :param dropout_rate: The dropout rate applied to various layers for regularization.
        :param activation: The activation function used within the encoder layers' feed-forward
            networks.
        """
        super(Encoder, self).__init__()
        self.pe = PositionalEncoding(d_model)
        self.embedding = nn.Embedding(input_vocab_size, d_model, padding_idx=0)
        self.dropout = nn.Dropout(dropout_rate)
        self.d_model = d_model

        self.encoder_layers = nn.ModuleList(
            [EncoderLayer(d_model, num_heads, dff, dropout_rate, activation)
             for _ in range(num_encoder_layers)]
        )

    def forward(
            self,
            x: torch.Tensor,
            src_mask: Optional[torch.Tensor] = None,
            src_key_padding_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Processes the input data through a series of transformer encoder layers after
        embedding it and applying positional encoding. The input is scaled by the
        square root of the model dimension prior to applying the dropout and
        positional encoding.

        :param x: Input tensor to be processed through the encoder model.
        :type x: torch.Tensor.
        :param src_mask: Tensor representing the source sequence mask, applied for
            masking specific tokens during self-attention computation. Defaults to None.
        :type src_mask: Optional[torch.Tensor]
        :param src_key_padding_mask: Tensor representing the key padding mask
            indicating which keys should be ignored in self-attention computation.
            Defaults to None.
        :type src_key_padding_mask: Optional[torch.Tensor]
        :return: Processed tensor after passing through all encoder layers.
        :rtype: torch.Tensor
        """
        x = self.embedding(x.to(torch.long)) * self.d_model ** 0.5
        x = self.dropout(self.pe(x))

        for layer in self.encoder_layers:
            x = layer(x, src_mask, src_key_padding_mask)

        return x


class Decoder(nn.Module):
    def __init__(
            self,
            num_decoder_layers: int,
            d_model: int,
            num_heads: int,
            dff: int,
            target_vocab_size: int,
            dropout_rate: float,
            activation: nn.Module = nn.ReLU()
    ) -> None:
        """
        A Decoder class that implements a multi-layer transformer decoder for sequence-to-sequence
        learning tasks. This class allows encoding of the target sequence, integrating positional
        encoding and multiple stacked decoder layers to enable complex transformations of input data.

        This implementation uses token embeddings combined with positional encoding, and processes
        the target sequence through a stack of decoder layers. It supports customizable activation
        functions within the decoder layers as well as configurable hyperparameters like the number
        of decoder layers, model dimensionality, attention heads, feed-forward network size, target
        vocabulary size, and dropout rate.

        :param num_decoder_layers: Number of decoder layers to stack in the model.
        :type num_decoder_layers: int
        :param d_model: Dimensionality of the model, defining the size of embedding space and key/query vectors.
        :type d_model: int
        :param num_heads: Number of attention heads used in the multi-head attention mechanism.
        :type num_heads: int
        :param dff: Dimensionality of the feed-forward network used within each decoder layer.
        :type dff: int
        :param target_vocab_size: The size of the vocabulary (number of distinct tokens) for the target language.
        :type target_vocab_size: int
        :param dropout_rate: The dropout rate applied for regularization in various parts of the decoder.
        :type dropout_rate: float
        :param activation: The activation function to use within the feed-forward sub-layer of the decoder.
        :type activation: nn.Module
        """
        super(Decoder, self).__init__()
        self.embedding = nn.Embedding(target_vocab_size, d_model, padding_idx=0)
        self.pe = PositionalEncoding(d_model)
        self.dropout = nn.Dropout(dropout_rate)
        self.d_model = d_model

        self.decoder_layers = nn.ModuleList(
            [DecoderLayer(d_model, num_heads, dff, dropout_rate, activation)
             for _ in range(num_decoder_layers)]
        )

    def forward(
            self,
            tgt: torch.Tensor,
            memory: torch.Tensor,
            tgt_mask: Optional[torch.Tensor] = None,
            memory_mask: Optional[torch.Tensor] = None,
            tgt_key_padding_mask: Optional[torch.Tensor] = None,
            memory_key_padding_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Processes the target sequence and memory using a transformer decoder. The method
        applies embedding and positional encoding to the target sequence, followed by
        iterative decoding through multiple decoder layers. Each decoder layer calculates
        self-attention and cross-attention weights, which are stored and returned alongside
        the final processed output.

        :param tgt: Target sequence to be decoded.
        :param memory: Memory sequence from the encoder to be attended during decoding.
        :param tgt_mask: Optional mask for the target sequence to control self-attention.
        :param memory_mask: Optional mask for the memory sequence to control cross-attention.
        :param tgt_key_padding_mask: Optional mask to ignore padding tokens in the target sequence.
        :param memory_key_padding_mask: Optional mask to ignore padding tokens in the encoder memory.
        :return: A tuple consisting of the processed target sequence and a dictionary of attention
            weights from all decoder layers.
        """
        attention_weights = {}

        x = self.embedding(tgt.to(torch.long)) * self.d_model ** 0.5
        x = self.dropout(self.pe(x))

        for i, layer in enumerate(self.decoder_layers):
            x, self_attention_weights, cross_attention_weights = layer(
                x, memory,
                tgt_mask=tgt_mask, memory_mask=memory_mask,
                tgt_key_padding_mask=tgt_key_padding_mask,
                memory_key_padding_mask=memory_key_padding_mask
            )

            attention_weights[
                "decoder_layer_{}_self_attention_weights".format(i + 1)
            ] = self_attention_weights
            attention_weights[
                "decoder_layer_{}_cross_attention_weights".format(i + 1)
            ] = cross_attention_weights

        return x, attention_weights


class Transformer(nn.Module):
    def __init__(
            self,
            num_encoder_layers: int,
            num_decoder_layers: int,
            d_model: int,
            num_heads: int,
            dff: int,
            input_vocab_size: int,
            target_vocab_size: int,
            dropout_rate: float,
            activation: nn.Module = nn.ReLU()
    ) -> None:
        """
        Initializes the Transformer model composed of an encoder and a decoder with
        configurable attributes. It provides mechanisms for sequence-to-sequence
        (Seq2Seq) tasks by transforming input sequences into output sequences
        based on attention mechanisms.

        :param num_encoder_layers: The number of layers in the encoder stack.
        :param num_decoder_layers: The number of layers in the decoder stack.
        :param d_model: Dimensionality of the attention embeddings and model.
        :param num_heads: The number of attention heads in multi-head attention layers.
        :param dff: Dimensionality of the feedforward intermediate layers.
        :param input_vocab_size: Size of the vocabulary for the input sequence data.
        :param target_vocab_size: Size of the vocabulary for the output sequence data.
        :param dropout_rate: Dropout rate applied during training for regularization.
        :param activation: Activation function module used in the feedforward network.
        """
        super(Transformer, self).__init__()

        self.encoder = Encoder(num_encoder_layers, d_model, num_heads,
                               dff, input_vocab_size, dropout_rate, activation)
        self.decoder = Decoder(num_decoder_layers, d_model, num_heads,
                               dff, target_vocab_size, dropout_rate, activation)
        self.output_fc = nn.Linear(d_model, target_vocab_size)

    def forward(
            self,
            src: torch.Tensor,
            tgt: torch.Tensor,
            src_mask: Optional[torch.Tensor] = None,
            tgt_mask: Optional[torch.Tensor] = None,
            memory_mask: Optional[torch.Tensor] = None,
            src_key_padding_mask: Optional[torch.Tensor] = None,
            tgt_key_padding_mask: Optional[torch.Tensor] = None,
            memory_key_padding_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Processes input and target sequences through an encoder-decoder architecture to
        generate logits and attention weights. The forward method integrates multiple
        transformer components such as an encoder, a decoder, and an output fully-connected
        layer. It also supports optional masking inputs for fine-tuned transformations
        and processing flexibility.

        :param src: Input tensor representing the source sequence.
        :param tgt: Input tensor representing the target sequence.
        :param src_mask: Optional tensor for source sequence masking.
        :param tgt_mask: Optional tensor for target sequence masking.
        :param memory_mask: Optional tensor for masking memory during decoding.
        :param src_key_padding_mask: Optional tensor for padding mask of source keys.
        :param tgt_key_padding_mask: Optional tensor for padding mask of target keys.
        :param memory_key_padding_mask: Optional tensor for padding mask of memory keys.
        :return: A tuple containing the logits tensor and a dictionary of attention weights.
        """
        memory = self.encoder(src, src_mask, src_key_padding_mask)
        decoder_output, attention_weights = self.decoder(
            tgt, memory, tgt_mask, memory_mask,
            tgt_key_padding_mask, memory_key_padding_mask)

        logits = self.output_fc(decoder_output)

        return logits, attention_weights