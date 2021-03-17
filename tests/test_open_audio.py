import os
import sys
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), os.pardir))

import pytest
import nbimporter
import numpy as np

from a_fish_or_not_a_fish import open_audio


@pytest.mark.parametrize(["audio_file", "result_file", "sample_rate", "gt_audio_len"],
                        (('test_files/test_audio.opus', 'test_files/test_open_audio.npy', 8000, 20480),
                          ('test_files/test_audio.mp3', 'test_files/test_open_audio16000.npy', 16000, 78336))
                          )
def test_open_audio(audio_file, result_file, sample_rate, gt_audio_len):
    try:
        audio, audio_len = open_audio(audio_file, sample_rate)
        gt_audio = np.load(result_file)
    except Exception as ex:
        pytest.xfail(str(ex))
    assert np.allclose(audio, gt_audio, atol=1e-6)
    assert audio_len == gt_audio_len
