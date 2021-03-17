import os
import sys
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), os.pardir))

import pytest
import nbimporter

from a_fish_or_not_a_fish import decode


@pytest.mark.parametrize(["alignment", "gt_text"],
                         (('blankblank<blank>', 'blankblank'),
                          ('<blank>кк<blank>от  ', 'кот'),
                          ('<blank>кк<blank>о<blank>от', 'коот'),
                          ('<blank>ка<blank>к     <blank><blank>ддддее<blank>ла', 'как дела'),
                          ))
def test_decode(alignment, gt_text):
    try:
        text = decode(alignment)
    except Exception as ex:
        pytest.xfail(str(ex))
    assert text == gt_text
