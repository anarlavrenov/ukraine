from torchtext import transforms as T


def _init_BERT_text_transform() -> T.Sequential:
    """
    Initializes a BERT-based text transformation pipeline using TorchText. This
    transformation applies the BERT tokenizer, converts string tokens to integers,
    appends special tokens for encoding ([CLS] and [SEP]), and finally converts the
    resulting sequence to a tensor format. This pipeline is specifically designed
    to prepare text data for BERT-based models.

    :return: A TorchText Sequential transformation that includes the BERT tokenizer,
             string-to-integer conversion, adding special tokens, and tensor conversion.
    :rtype: torchtext.transforms.Sequential
    """
    return T.Sequential(
        T.BERTTokenizer(
            vocab_path="https://huggingface.co/bert-base-cased/resolve/main/vocab.txt",
            do_lower_case=True,
            return_tokens=False
        ),
        T.StrToIntTransform(),
        T.AddToken(101, begin=True),
        T.AddToken(102, begin=False),
        T.ToTensor()
    )
