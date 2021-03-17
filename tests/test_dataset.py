import os
import sys
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), os.pardir))

import torch
import pytest
import nbimporter

from vocab import vocab
from a_fish_or_not_a_fish import AudioDataset


@pytest.mark.parametrize(["i", "audio_len", "text", "tokens"],
                         ((0, 39168, "тестовое аудио", [19,  5, 18, 19, 15,  2, 15,  5, 33,  0, 20,  4,  9, 15]),
                         (1, 20480, "и это тоже", [9, 33, 30, 19, 15, 33, 19, 15,  7,  5])
                          ))
def test_dataset(i, audio_len, text, tokens):
    try:
        dataset = AudioDataset('test_files/test_dataset.txt', vocab)
    except Exception as ex:
        pytest.xfail(str(ex))
    assert len(dataset) == 2
    assert dataset[i]["audio"].shape == torch.Size([audio_len])
    assert dataset[i]["audio_len"] == audio_len
    assert dataset[i]["text"] == text
    assert dataset[i]["text_len"] == len(text)
    assert (dataset[i]["tokens"].numpy() == tokens).all()
