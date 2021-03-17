import os
import sys
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), os.pardir))

import pytest
import nbimporter

from a_fish_or_not_a_fish import calc_wer


@pytest.mark.parametrize(["prediction", "gt_test", "gt_wer"],
                         (('тинькофф', 'тинькофф', 0),
                          ('оператор александр тинькофф банк', 'оператор алексей тинькофф банк', 0.25),
                          ('оператор александр тинькофф банк', 'вам звонит оператор алексей тинькофф банк', 0.5),
                          ('оператор александр тинькофф банк', 'тинькофф банк', 1.0),
                          ('оператор александр тинькофф банк', '', 1.),
                          ('', '', 0),
                          ('', 'тинькофф банк', 1.)
                          ))
def test_compute_wer(prediction, gt_test, gt_wer):
    try:
        wer = calc_wer(prediction, gt_test)
    except Exception as ex:
        pytest.xfail(str(ex))
    assert wer == gt_wer
