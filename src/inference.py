import os

import torch

from src.deepspeech import Model
from src.datasets import collate_fn
from src.optimization import get_prediction
from src.decoding import fast_beam_search_decode
from vocabulary import Vocab, get_blank_index, get_num_tokens
from src.audio_utils import open_audio, compute_log_mel_spectrogram


def load_from_ckpt(_model, ckpt_path):
    checkpoint = torch.load(ckpt_path, map_location='cpu')
    _model.load_state_dict(checkpoint['model_state_dict'])


class InferenceModel:
    _alphabet = [
        'а', 'б', 'в', 'г', 'д', 'е', 'ё', 'ж', 'з', 'и', 'й', 'к',
        'л', 'м', 'н', 'о', 'п', 'р', 'с', 'т', 'у', 'ф', 'х', 'ц',
        'ч', 'ш', 'щ', 'ь', 'ы', 'ъ', 'э', 'ю', 'я', ' ', '<blank>'
    ]

    def __init__(
            self, checkpoint_path='/home/mnakhodnov/sirius-stt/models/6_recovered_v3/epoch_0.pt',
            device=torch.device('cpu')
    ):
        if not os.path.exists(checkpoint_path):
            raise ValueError(f'There is no checkpoint in {checkpoint_path}')

        self.device = device
        self.checkpoint_path = checkpoint_path

        self._vocab = Vocab(self._alphabet)

        self._num_tokens = get_num_tokens(self._vocab)
        self._blank_index = get_blank_index(self._vocab)

        self._sample_rate = 8000
        self._model_config = {
            'num_mel_bins': 64,
            'hidden_size': 512,
            'num_layers': 4,
            'num_tokens': len(self._vocab.tokens2indices()) - 1,
        }

        self.model = Model(**self._model_config)
        load_from_ckpt(self.model, self.checkpoint_path)
        self.model.eval()

        self.decoder = fast_beam_search_decode
        # self._kenlm_binary_path = '/data/mnakhodnov/language_data/cc100/xaa.processed.4.binary'
        self._kenlm_binary_path = '/data/mnakhodnov/language_data/common_voice/train.txt.binary'
        self.decoder_kwargs = {
            'beam_size': 100, 'cutoff_top_n': 10, 'cutoff_prob': 1.0,
            'ext_scoring_func': self._kenlm_binary_path, 'alpha': 2.0, 'beta': 0.3, 'num_processes': 32
        }

    def run(self, audio_path):
        with torch.no_grad():

            audio, audio_len = open_audio(audio_path, desired_sample_rate=self._sample_rate)
            batch = collate_fn([
                [audio, audio_len, '', 0, torch.tensor([])]
            ])
            audios, audio_lens, texts, text_lens, tokens, *_ = batch
            batch = {
                'audios': audios,
                'audio_lens': audio_lens,
                'texts': texts,
                'text_lens': text_lens,
                'tokens': tokens,
            }

            batch = {
                key: value.to(device=self.device) if isinstance(value, torch.Tensor) else value
                for key, value in batch.items()
            }

            log_mel_spectrogram, seq_lens = compute_log_mel_spectrogram(
                audio=batch['audios'], sequence_lengths=batch['audio_lens'], spectrogram_transform=None
            )
            logprobs, seq_lens = self.model(log_mel_spectrogram, seq_lens)

            predictions = get_prediction(
                logprobs, seq_lens, self._vocab, decoder=self.decoder, decoder_kwargs=self.decoder_kwargs
            )
            return predictions[0]
