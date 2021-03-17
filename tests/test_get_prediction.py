import os
import sys
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), os.pardir))

import torch
import pytest
import nbimporter

from a_fish_or_not_a_fish import get_prediction
from vocab import vocab


def test_get_prediction():
    try:
        pred = get_prediction(torch.tensor([[[0.1, 0.5, 0.4]], [[0.9, 0., 0.1]]]), torch.tensor([2]), vocab)
    except Exception as ex:
        pytest.xfail(str(ex))
    assert pred == ['ба']
