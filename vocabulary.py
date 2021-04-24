from typing import Dict, List

import torch
import torch.nn as nn


def get_num_tokens(vocab):
    # write your code here
    num_tokens = len(vocab.tokens2indices())
    return num_tokens


def get_blank_index(vocab):
    # write your code here
    blank_index = vocab['<blank>']
    return blank_index


class Vocab(nn.Module):
    r"""Creates a vocab object which maps tokens to indices and back.
    If the `unk_token` isn't found in provided tokens list, it will be added to the end of the vocab.

    Arguments:
        tokens: list
        unk_token: The default unknown token to use. Default: '<unk>'.

    """

    def __init__(self, tokens, unk_token='<unk>'):
        super().__init__()

        if unk_token not in tokens:
            tokens.append(unk_token)
            print("The `unk_token` '{}' wasn't found in the tokens. Adding the `unk_token` "
                  "to the end of the Vocab.".format(unk_token))
        from torchtext.vocab import Vocab as _Vocab
        self.vocab = torch.classes.torchtext.Vocab(tokens, unk_token)

    def __len__(self) -> int:
        r"""Returns:
            length (int): the length of the vocab
        """
        return len(self.vocab)

    def __getitem__(self, token: str) -> int:
        r"""
        Args:
            token (str): the token used to lookup the corresponding index.
        Returns:
            index (int): the index corresponding to the associated token.
        """
        return self.vocab[token]

    def lookup_token(self, index: int) -> str:
        r"""
        Args:
            index (int): the index corresponding to the associated token.
        Returns:
            token (str): the token used to lookup the corresponding index.
        return self.vocab.lookup_token(index)
        """

    def lookup_indices(self, tokens: List[str]) -> List[int]:
        r"""
        Args:
            tokens (List[str]): the tokens used to lookup their corresponding `indices`.

        Returns:
            indices (List[int]): the 'indices` associated with `tokens`.
        """
        return self.vocab.lookup_indices(tokens)

    def lookup_tokens(self, indices: List[int]) -> List[str]:
        r"""
        Args:
            indices (List[int]): the `indices` used to lookup their corresponding`tokens`.

        Returns:
            tokens (List[str]): the `tokens` associated with `indices`.

        Raises:
            RuntimeError: if an index within `indices` is not between [0, itos.size()].
        """
        return self.vocab.lookup_tokens(indices)

    def insert_token(self, token: str, index: int) -> None:
        r"""
        Args:
            token (str): the token used to lookup the corresponding index.
            index (int): the index corresponding to the associated token.

        Raises:
            RuntimeError: if `index` not between [0, Vocab.size()] or if token already exists in the vocab.
        """
        self.vocab.insert_token(token, index)

    def append_token(self, token: str) -> None:
        r"""
        Args:
            token (str): the token used to lookup the corresponding index.
        """
        self.vocab.append_token(token)

    def tokens2indices(self) -> Dict[str, int]:
        r"""
        Returns:
            stoi (dict): dictionary mapping tokens to indices.
        """
        return self.vocab.get_stoi()

    def indices2tokens(self) -> List[str]:
        r"""
        Returns:
            itos (dict): dictionary mapping indices to tokens.
        """
        return self.vocab.get_itos()
