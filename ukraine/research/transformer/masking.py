import torch
from typing import Tuple


def generate_square_subsequent_mask(size: int, device: str) -> torch.Tensor:
    """
    Generate a mask tensor for subsequent square-shaped elements. This is typically
    used in tasks like language modeling, where the future tokens should not
    be visible to the model during training or inference. The generated mask blocks
    future positions, allowing only access to previous tokens and the current one.

    The mask is represented as a 2D tensor of shape `(size, size)`, where cells below
    the main diagonal are filled with zeros, and cells above the diagonal are filled
    with negative infinity (`-inf`).

    :param device: A string: 'cuda' or 'cpu'.
    :param size: The size of the square mask. Represents both the number of rows
        and columns of the mask tensor.
    :return: A 2D tensor of shape `(size, size)` where the upper triangular part
        above the main diagonal is filled with negative infinity (`-inf`), and the
        rest is filled with zeros.
    :rtype: torch.Tensor
    """
    return torch.triu(torch.ones((size, size)) * float("-inf"), diagonal=1).to(device)


# noinspection PyTypeChecker
def generate_padding_mask(
        lengths: torch.Tensor,
        max_seq_len: int,
        device: str
) -> torch.Tensor:
    """
    Generates a binary padding mask for sequences with varying lengths. The mask identifies positions
    in the sequences that should be treated as padding based on the given lengths and maximum sequence
    length. Each row in the resulting mask corresponds to a sequence, where a value of `True` signifies
    a padding position and `False` signifies a valid position.

    :param device: A string: 'cuda' or 'cpu'.
    :param lengths: A tensor of shape ``(batch_size,)`` containing the actual lengths of each sequence
        in the batch. All values represent the number of valid (non-padded) tokens for each sequence.
    :param max_seq_len: An integer representing the maximum length of sequences in the batch. This is
        used to define the width of the resulting mask.
    :return: A binary tensor of shape ``(batch_size, max_seq_len)``, where each row represents the
        padding mask for the corresponding sequence in the batch. The values are ``True`` for padding
        positions and ``False`` for valid positions.
    """
    batch_size = lengths.size(0)
    ids = torch.arange(0, max_seq_len).unsqueeze(0).expand(batch_size, -1).to(device)
    return ids >= lengths.unsqueeze(1).expand(batch_size, max_seq_len).to(device)


def generate_masks(
        tgt: torch.Tensor,
        src_lengths: torch.Tensor,
        tgt_lengths: torch.Tensor,
        max_seq_len_src: int,
        max_seq_len_tgt: int,
        device: str
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Generates the necessary masks for processing sequences in tasks such as
    sequence-to-sequence models. This includes a subsequent mask for the target
    sequence to prevent access to future tokens, and padding masks for both source
    and target sequences.

    :param tgt: Target sequence tensor.
    :param src_lengths: Tensor containing the lengths of each sequence in the
        source batch.
    :param tgt_lengths: Tensor containing the lengths of each sequence in the
        target batch.
    :param max_seq_len_src: Maximum sequence length of the source batch.
    :param max_seq_len_tgt: Maximum sequence length of the target batch.
    :param device: String representing the device on which the tensors will
        reside (e.g., 'cpu' or 'cuda').
    :return: A tuple containing three tensors: the target mask, source key padding
        mask, and target key padding mask.
    """
    tgt_mask = generate_square_subsequent_mask(tgt.size(1), device=device)
    src_key_padding_mask = generate_padding_mask(
        src_lengths,
        max_seq_len_src,
        device=device
    )
    tgt_key_padding_mask = generate_padding_mask(
        tgt_lengths,
        max_seq_len_tgt,
        device=device
    )

    return (
        tgt_mask,
        src_key_padding_mask,
        tgt_key_padding_mask
    )