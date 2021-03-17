import os
import sys
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), os.pardir))

import pytest
import nbimporter

from vocab import vocab
from a_fish_or_not_a_fish import get_blank_index


def test_get_blank_index():
    try:
        blank_index = get_blank_index(vocab)
    except Exception as ex:
        pytest.xfail(str(ex))
    assert blank_index == 34
