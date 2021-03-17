import os
import sys
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), os.pardir))

import pytest
import nbimporter

from vocab import vocab
from a_fish_or_not_a_fish import get_num_tokens


def test_get_num_tokens():
    try:
        num_tokens = get_num_tokens(vocab)
    except Exception as ex:
        pytest.xfail(str(ex))
    assert num_tokens == 35 or num_tokens == 36
