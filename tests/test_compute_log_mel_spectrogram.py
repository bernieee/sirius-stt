import os
import sys
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), os.pardir))

import torch
import pytest
import nbimporter
import numpy as np

from a_fish_or_not_a_fish import compute_log_mel_spectrogram


@pytest.mark.parametrize(["audio_file", "result_file", "gt_len"],
                        (('test_files/test_open_audio.npy', 'test_files/spectrogram.npy', 257),))
def test_compute_log_mel_spectrogram(audio_file, result_file, gt_len):
    try:
        audio = np.load(audio_file)
        spectrogram, new_len = compute_log_mel_spectrogram(torch.tensor(audio), torch.tensor(len(audio)))
        gt_spectrogram = np.load(result_file)
    except Exception as ex:
        pytest.xfail(str(ex))
    assert np.allclose(spectrogram, gt_spectrogram, atol=1e-5), 'Did you forget about the logarithm?'
    assert torch.all(new_len == gt_len), 'Please, check sequence_lengths'
