import os
import sys
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), os.pardir))

import torch
import pytest
import nbimporter
import torch.nn as nn

from a_fish_or_not_a_fish import Model


def test_layers():
    model = Model(num_mel_bins=64, hidden_size=512, num_layers=4, num_tokens=35)
    layers = [layer for layer in model.modules()][2:]
    gt_types = [nn.Conv2d, nn.BatchNorm2d, nn.ReLU, nn.Conv2d, nn.BatchNorm2d, nn.ReLU, nn.LSTM, nn.Linear]
    error_message = ''

    for i, (layer, gt_type) in enumerate(zip(layers, gt_types)):
        if not isinstance(layer, gt_type):
            error_message = f"Please, check your {i + 1} layer: its type should be {gt_type}, but it is {type(layer)}"
            break

        if isinstance(layer, nn.Conv2d) and layer.bias is not None:
            error_message = f"Please, set bias=False for conv layers (layer number {i + 1})"
            break

        elif isinstance(layer, nn.BatchNorm2d) and not layer.momentum==0.9:
            error_message = f"Please, set momentum==0.9 for BatchNorm layers (layer number {i + 1})"
            break
    del model
    if error_message:
        raise ValueError(error_message)


@pytest.mark.parametrize(["old_seq_lens", "new_seq_lens"],
                         ((100, 27),
                          (503, 161)
                          ))
def test_seq_lens(old_seq_lens, new_seq_lens):
    new_seq_lens_ = int(Model.get_new_seq_lens(torch.tensor(old_seq_lens), 11, 1, 11, 3))
    assert new_seq_lens_ == new_seq_lens


def test_transpose_and_reshape():
    result = Model.transpose_and_reshape(torch.tensor([[[[1, 3]]]])).shape
    assert result == torch.Size([1, 2, 1])


def test_forward():
    model = Model(num_mel_bins=64, hidden_size=512, num_layers=4, num_tokens=35)
    outputs, seq_lens = model(torch.rand([3, 64, 1000]), torch.tensor([1000, 555, 900]))
    del model
    assert outputs.shape == torch.Size([327, 3, 35])
    assert list(seq_lens) == list(torch.tensor([327, 179, 294]))
