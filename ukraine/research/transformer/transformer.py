import torch
from torch import nn
from .layers import (
    PositionalEncoding,
    MultiheadAttention
)
from typing import Tuple, Optional, Dict, Callable


class EncoderLayer(nn.Module):
    """
    Represents a single layer of a Transformer encoder.

    This class implements the functionality of a Transformer encoder
    layer, which consists of a multi-head attention mechanism followed
    by a normalization step, dropout, and a feed-forward module.
    It is designed to process input sequences and produce encoded
    representations, which can be further passed through additional
    layers for tasks such as language modeling or sequence-to-sequence
    translation.

    :ivar d_model: Dimensionality of the model, representing the size of the
        input and output of the layer.
    :type d_model: int
    :ivar num_heads: Number of attention heads used in the multi-head
        attention mechanism.
    :type num_heads: int
    :ivar mha: The multi-head attention module handling the attention
        operations within the layer.
    :type mha: MultiheadAttention
    :ivar ff_module: The feed-forward module applied after the multi-head
        attention block, which typically consists of linear layers.
    :type ff_module: nn.Module
    :ivar norm1: Normalization module applied after the multi-head
        attention block and dropout.
    :type norm1: nn.Module
    :ivar norm2: Normalization module applied after the feed-forward
        block.
    :type norm2: nn.Module
    :ivar dropout: Dropout layer applied to the output of the multi-head
        attention mechanism to prevent overfitting.
    :type dropout: nn.Dropout
    """
    def __init__(
            self,
            d_model: int,
            num_heads: int,
            dropout_rate: float,
            ff_factory: Callable[[], nn.Module],
            norm_factory: Callable[[], nn.Module],
            use_flash: bool = False,
    ) -> None:
        super(EncoderLayer, self).__init__()

        self.d_model = d_model
        self.num_heads = num_heads
        self.mha = MultiheadAttention(d_model, num_heads, dropout_rate, use_flash=use_flash)

        self.ff_module = ff_factory()

        self.norm1 = norm_factory()
        self.norm2 = norm_factory()
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self,
                x: torch.Tensor,
                mask: Optional[torch.Tensor] = None,
                src_key_padding_mask: Optional[torch.Tensor] = None
                ) -> torch.Tensor:
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

        nx = self.norm1(x)
        attention_output, attention_weights = self.mha(
            nx, nx, nx, mask, src_key_padding_mask)
        out1 = x + self.dropout(attention_output)
        out2 = out1 + self.ff_module(self.norm2(out1))

        return out2


class DecoderLayer(nn.Module):
    """
    Represents a single decoder layer in a Transformer-based model architecture.

    This class defines a Transformer decoder layer, which performs sequential
    processing on the input data using self-attention, cross-attention, and a feedforward
    module. Each stage of the decoder layer applies normalization and dropout to enable
    efficient learning and regularization. The decoder layer is designed to handle
    target sequences and information retrieved from an encoder via embedded representations.

    :ivar self_attention: Multi-head self-attention mechanism for processing the target
        sequences.
    :type self_attention: MultiheadAttention
    :ivar cross_attention: Multi-head cross-attention mechanism for processing the
        encoder memory and target sequences.
    :type cross_attention: MultiheadAttention
    :ivar ff_module: Feedforward module applied after the attention mechanisms.
    :type ff_module: nn.Module
    :ivar dropout1: Dropout module applied after the self-attention mechanism.
    :type dropout1: nn.Dropout
    :ivar dropout2: Dropout module applied after the cross-attention mechanism.
    :type dropout2: nn.Dropout
    :ivar norm1: Normalization module applied after self-attention and subsequent
        dropout.
    :type norm1: nn.Module
    :ivar norm2: Normalization module applied after cross-attention and subsequent
        dropout.
    :type norm2: nn.Module
    :ivar norm3: Normalization module applied after the feedforward module.
    :type norm3: nn.Module
    """
    def __init__(
            self,
            d_model: int,
            num_heads: int,
            dropout_rate: float,
            ff_factory: Callable[[], nn.Module],
            norm_factory: Callable[[], nn.Module],
            use_flash: bool = False,
            use_cross_attn: bool = False,
            cross_fusion: bool = False
    ):
        super(DecoderLayer, self).__init__()

        self.self_attention = MultiheadAttention(d_model, num_heads, dropout_rate, use_flash=use_flash)
        self.ff_module = ff_factory()

        self.dropout1 = nn.Dropout(dropout_rate)
        self.norm1 = norm_factory()
        self.norm3 = norm_factory()

        self.cross_fusion = cross_fusion
        self.use_cross_attn = use_cross_attn

        if self.cross_fusion and not self.use_cross_attn:
            raise ValueError("cross_fusion=True requires use_cross_attn=True")

        if self.use_cross_attn:
            self.cross_attention = MultiheadAttention(d_model, num_heads, dropout_rate, use_flash=use_flash)
            self.cross_fc = nn.Linear(2 * d_model, d_model) if self.cross_fusion else None
            self.dropout2 = nn.Dropout(dropout_rate)
            self.norm2 = norm_factory()
        else:
            self.cross_attention = None
            self.cross_fc = None
            self.dropout2 = None
            self.norm2 = None

    def forward(
            self,
            x: torch.Tensor,
            memory: Optional[torch.Tensor] = None,
            tgt_mask: Optional[torch.Tensor] = None,
            memory_mask: Optional[torch.Tensor] = None,
            tgt_key_padding_mask: Optional[torch.Tensor] = None,
            memory_key_padding_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
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

        nx = self.norm1(x)
        self_attention_output, self_attention_weights = self.self_attention(
            nx, nx, nx,
            mask=tgt_mask, key_padding_mask=tgt_key_padding_mask)
        out1 = x + self.dropout1(self_attention_output)

        if self.use_cross_attn and memory is not None:
            cross_attention_output, cross_attention_weights = self.cross_attention(
                self.norm2(out1), memory, memory, mask=memory_mask, key_padding_mask=memory_key_padding_mask)
            if self.cross_fusion:
                assert self.cross_fc is not None
                concat = torch.cat((out1, cross_attention_output), dim=-1)
                out2 = out1 + self.dropout2(self.cross_fc(concat))
            else:
                out2 = out1 + self.dropout2(cross_attention_output)
        else:
            cross_attention_weights = None
            out2 = out1

        out3 = out2 + self.ff_module(self.norm3(out2))

        return out3, self_attention_weights, cross_attention_weights


class Encoder(nn.Module):
    """
    Defines the Transformer Encoder component as used in sequence-to-sequence
    models. This class is responsible for encoding input sequences into an
    intermediate representation using attention mechanisms and feed-forward
    neural networks. Each layer within the encoder applies self-attention,
    positional encoding, feed-forward transformations, and normalization.

    :ivar pe: The positional encoding module responsible for adding positional
        information to the embedding space.
    :type pe: PositionalEncoding
    :ivar embedding: Embedding module which converts input indices into dense
        vectors of dimension `d_model`.
    :type embedding: nn.Embedding
    :ivar dropout: Dropout layer to regularize the encoder output by applying
        dropout to specific elements.
    :type dropout: nn.Dropout
    :ivar d_model: Dimension of the embedding and internal representations within
        the encoder layers.
    :type d_model: int
    :ivar encoder_layers: Sequential module list containing the individual
        encoder layers as specified by the `num_encoder_layers` parameter.
    :type encoder_layers: nn.ModuleList
    """
    def __init__(
            self,
            num_encoder_layers: int,
            d_model: int,
            num_heads: int,
            input_vocab_size: int,
            dropout_rate: float,
            ff_factory: Callable[[], nn.Module],
            norm_factory: Callable[[], nn.Module],
            pad_token_id: int,
            use_flash: bool = False
    ) -> None:
        """
        Initializes the Encoder with specified parameters and creates required
        attributes such as positional encoding, embedding, and encoder layers
        using the provided configurations. This class sets up the building blocks
        for a Transformer Encoder module.

        :param num_encoder_layers: The number of encoder layers in the model.
        :param d_model: The dimension of the embedding space and internal model representation.
        :param num_heads: The number of attention heads in the multi-head attention mechanism.
        :param input_vocab_size: The size of the input vocabulary for the embedding layer.
        :param dropout_rate: The dropout rate applied to various layers for regularization.
        """
        super(Encoder, self).__init__()
        self.pe = PositionalEncoding(d_model)
        self.embedding = nn.Embedding(input_vocab_size, d_model, padding_idx=pad_token_id)
        self.dropout = nn.Dropout(dropout_rate)
        self.d_model = d_model

        self.encoder_layers = nn.ModuleList(
            [EncoderLayer(d_model, num_heads, dropout_rate, ff_factory, norm_factory, use_flash=use_flash)
             for _ in range(num_encoder_layers)]
        )
        self.final_layer_norm = norm_factory()

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
        x = self.embedding(x.to(torch.long)) * (self.d_model ** 0.5)
        x = self.dropout(self.pe(x))

        for layer in self.encoder_layers:
            x = layer(x, src_mask, src_key_padding_mask)
        x = self.final_layer_norm(x)

        return x


class Decoder(nn.Module):
    """
    A Decoder class that implements a multi-layer transformer decoder for sequence-to-sequence
    learning tasks. This class allows encoding of the target sequence, integrating positional
    encoding and multiple stacked decoder layers to enable complex transformations of input data.

    This implementation uses token embeddings combined with positional encoding, and processes
    the target sequence through a stack of decoder layers. It supports customizable activation
    functions within the decoder layers as well as configurable hyperparameters like the number
    of decoder layers, model dimensionality, attention heads, feed-forward network size, target
    vocabulary size, and dropout rate.

    :ivar embedding: Embedding layer to convert token indices to dense vectors.
    :type embedding: nn.Embedding
    :ivar pe: Positional encoding module providing positional information to tokens.
    :type pe: PositionalEncoding
    :ivar dropout: Dropout layer for regularization purposes.
    :type dropout: nn.Dropout
    :ivar d_model: Dimensionality of the model, defining the embedding space size.
    :type d_model: int
    :ivar decoder_layers: Module list containing stacked decoder layers.
    :type decoder_layers: nn.ModuleList
    """
    def __init__(
            self,
            num_decoder_layers: int,
            d_model: int,
            num_heads: int,
            target_vocab_size: int,
            dropout_rate: float,
            ff_factory: Callable[[], nn.Module],
            norm_factory: Callable[[], nn.Module],
            pad_token_id: int,
            use_flash: bool = False,
            use_cross_attn: bool = False,
            cross_fusion: bool = False
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
        :param target_vocab_size: The size of the vocabulary (number of distinct tokens) for the target language.
        :type target_vocab_size: int
        :param dropout_rate: The dropout rate applied for regularization in various parts of the decoder.
        :type dropout_rate: float
        """
        super(Decoder, self).__init__()
        self.embedding = nn.Embedding(target_vocab_size, d_model, padding_idx=pad_token_id)
        self.pe = PositionalEncoding(d_model)
        self.dropout = nn.Dropout(dropout_rate)
        self.d_model = d_model

        self.decoder_layers = nn.ModuleList(
            [DecoderLayer(d_model, num_heads, dropout_rate, ff_factory,
                          norm_factory, use_flash=use_flash, use_cross_attn=use_cross_attn, cross_fusion=cross_fusion)
             for _ in range(num_decoder_layers)]
        )
        self.final_layer_norm = norm_factory()

    def forward(
            self,
            tgt: torch.Tensor,
            memory: Optional[torch.Tensor] = None,
            tgt_mask: Optional[torch.Tensor] = None,
            memory_mask: Optional[torch.Tensor] = None,
            tgt_key_padding_mask: Optional[torch.Tensor] = None,
            memory_key_padding_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Dict[str, Optional[torch.Tensor]]]:
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

        x = self.embedding(tgt.to(torch.long)) * (self.d_model ** 0.5)
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

        x = self.final_layer_norm(x)

        return x, attention_weights


class Transformer(nn.Module):
    """
    Defines a Transformer model, combining encoder and decoder modules with attention
    mechanisms for sequence-to-sequence tasks. This class is designed to transform input
    sequences into output sequences efficiently through multi-head attention and fully
    connected layers. The architecture supports configurable components, including the
    number of encoder/decoder layers, attention heads, and custom feedforward/norm modules.

    :ivar encoder: Encoder module for the input sequence transformation.
    :type encoder: Encoder
    :ivar decoder: Decoder module for generating target sequences.
    :type decoder: Decoder
    :ivar output_fc: Fully connected layer projecting decoder outputs to target
        vocabulary size.
    :type output_fc: torch.nn.Linear
    """
    def __init__(
            self,
            num_encoder_layers: int,
            num_decoder_layers: int,
            d_model: int,
            num_heads: int,
            input_vocab_size: int,
            target_vocab_size: int,
            dropout_rate: float,
            ff_factory: Callable[[], nn.Module],
            norm_factory: Callable[[], nn.Module],
            pad_token_id: int,
            use_encoder: bool = True,
            use_flash: bool = False,
            cross_fusion: bool = False
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
        :param input_vocab_size: Size of the vocabulary for the input sequence data.
        :param target_vocab_size: Size of the vocabulary for the output sequence data.
        :param dropout_rate: Dropout rate applied during training for regularization.
        """
        super(Transformer, self).__init__()

        self.use_encoder = use_encoder
        if cross_fusion and not use_encoder:
            raise ValueError("cross_fusion=True requires use_encoder=True.")

        if use_encoder:
            self.encoder = Encoder(num_encoder_layers, d_model, num_heads,
                                   input_vocab_size, dropout_rate,
                                   ff_factory, norm_factory,
                                   pad_token_id, use_flash=use_flash)
        else:
            self.encoder = None

        self.decoder = Decoder(num_decoder_layers, d_model, num_heads,
                               target_vocab_size, dropout_rate,
                               ff_factory, norm_factory,
                               pad_token_id, use_flash=use_flash,
                               use_cross_attn=self.use_encoder, cross_fusion=cross_fusion)
        self.output_fc = nn.Linear(d_model, target_vocab_size, bias=False)

    def forward(
            self,
            src: Optional[torch.Tensor] = None,
            tgt: Optional[torch.Tensor] = None,
            src_mask: Optional[torch.Tensor] = None,
            tgt_mask: Optional[torch.Tensor] = None,
            memory_mask: Optional[torch.Tensor] = None,
            src_key_padding_mask: Optional[torch.Tensor] = None,
            tgt_key_padding_mask: Optional[torch.Tensor] = None,
            memory_key_padding_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Dict[str, Optional[torch.Tensor]]]:
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

        if self.use_encoder and src is not None:
            memory = self.encoder(src, src_mask, src_key_padding_mask)
            decoder_input = tgt
        else:
            memory = None
            decoder_input = src if tgt is None else tgt

        if decoder_input is None:
            raise ValueError("Either source or target sequence must be provided.")

        decoder_output, attention_weights = self.decoder(
            decoder_input, memory, tgt_mask, memory_mask,
            tgt_key_padding_mask, memory_key_padding_mask)

        logits = self.output_fc(decoder_output)

        return logits, attention_weights
