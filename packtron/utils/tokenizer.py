# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.

# This file is derived from Megatron-LM and has been modified for PackTron.

"""PackTron tokenizers."""

import json
from collections import OrderedDict
import transformers

def build_tokenizer(args, **kwargs):
    """Initialize tokenizer."""
    # Select and instantiate the tokenizer.
    tokenizer = _HuggingFaceTokenizer(args.tokenizer_model, **kwargs)

    return tokenizer


class _HuggingFaceTokenizer():
    def __init__(self, pretrained_model_name_or_path, **kwargs):
        # Initialize unique_identifiers for dataset caching (required by MegatronDataset)
        self.unique_identifiers = OrderedDict()
        self.unique_identifiers["class"] = type(self).__name__
        self.unique_identifiers["tokenizer_path"] = [pretrained_model_name_or_path]
        for option in kwargs:
            self.unique_identifiers[option] = str(kwargs[option])
        self.unique_description = json.dumps(self.unique_identifiers, indent=4)
        
        self._tokenizer = transformers.AutoTokenizer.from_pretrained(
            pretrained_model_name_or_path=pretrained_model_name_or_path, **kwargs
        )
        self._vocab = self._tokenizer.get_vocab()
        self._inv_vocab = {token_id: token for token, token_id in self._vocab.items()}

    @property
    def vocab_size(self):
        return len(self._tokenizer)

    @property
    def vocab(self):
        """Dictionary from vocab text token to id token."""
        return self._vocab

    @property
    def inv_vocab(self):
        """Dictionary from vocab id token to text token."""
        return self._inv_vocab

    @property
    def decoder(self):
        return self._inv_vocab

    def tokenize(self, text, **kwargs):
        return self._tokenizer(text, **kwargs).input_ids

    def detokenize(self, token_ids, **kwargs):
        return self._tokenizer.decode(token_ids, **kwargs)

    def offsets(self, ids: list[int], text: str) -> list[int]:
        retok_ids: "transformers.BatchEncoding" = self._tokenizer(text)
        offsets, next_start_idx = [], 0
        for i in range(len(ids)):
            span = retok_ids.token_to_chars(i)
            if span is not None:
                offsets.append(span.start)
                next_start_idx = span.end
            else:
                offsets.append(next_start_idx)
        return offsets

    @property
    def eod(self):
        return self._tokenizer.eos_token_id

    @property
    def bos(self):
        return self._tokenizer.bos_token_id

def _add_tokenizer_args(parser):
    group = parser.add_argument_group(title='tokenizer')
    group.add_argument('--vocab-size', type=int, default=None,
                       help='Size of vocab before EOD or padding.')
    group.add_argument('--vocab-file', type=str, default=None,
                       help='Path to the vocab file.')
    group.add_argument('--merge-file', type=str, default=None,
                       help='Path to the BPE merge file.')
    group.add_argument('--vocab-extra-ids', type=int, default=0,
                       help='Number of additional vocabulary tokens. '
                            'They are used for span masking in the T5 model')
    group.add_argument('--tokenizer-model', type=str, default=None,
                       help='Sentencepiece tokenizer model.')
    return parser
