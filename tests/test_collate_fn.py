import os
import sys
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), os.pardir))

import torch
import pytest
import nbimporter

from vocab import vocab
from a_fish_or_not_a_fish import AudioDataset, collate_fn


def test_collate_fn():
    try:
        dataset = AudioDataset('test_files/test_dataset.txt', vocab)
    except Exception as ex:
        pytest.xfail(str(ex))
    try:
        batch = collate_fn([dataset[i] for i in [0, 1]])
    except Exception as ex:
        pytest.xfail(str(ex))

    assert batch["audios"].shape == torch.Size([2, 39168])
    assert (batch["audio_lens"].numpy() == [39168, 20480]).all()
    assert batch["texts"] == ["тестовое аудио", "и это тоже"]
    assert (batch["text_lens"].numpy() == [14, 10]).all()
    assert batch["tokens"].shape == torch.Size([2, 14])
