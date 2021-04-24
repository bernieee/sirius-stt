import numpy as np

import torch
import torchaudio


def open_audio(audio_path, desired_sample_rate, effects=None):
    """ Open and resample audio, average across channels
        Inputs:

            audio_path: str, path to audio
            desired_sample_rate: int, the sampling rate to which we would like to convert the audio
        Returns:
            audio: 1D tensor with shape (num_timesteps)
            audio_len: int, len of audio
    """
    # write your code here
    if effects is None:
        # noinspection PyUnresolvedReferences
        audio_data, orig_sample_rate = torchaudio.load(audio_path, channels_first=True)
        resampler = torchaudio.transforms.Resample(orig_freq=orig_sample_rate, new_freq=desired_sample_rate)
        audio_data = resampler(audio_data)
    else:
        # Sampling method is inconsistent with torchaudio.transforms.Resample
        audio_data, _ = torchaudio.sox_effects.apply_effects_file(
            audio_path, effects=[*effects, ['rate', str(desired_sample_rate)]], channels_first=True
        )

    audio_data = torch.mean(audio_data, dim=0)
    return audio_data, audio_data.shape[0]


class AudioTransformsChain:
    def __init__(self):
        self.probas = []
        self.transforms_chain = []

    def add_group(self, proba, transforms):
        self.probas.append(proba)
        self.transforms_chain.append(transforms)
        return self

    def sample(self):
        sampled_effects = []
        for proba, transforms_group in zip(self.probas, self.transforms_chain):
            if np.random.uniform(0.0, 1.0) > proba:
                continue
            for transform in transforms_group:
                sampled_effects += transform.sample()
        return sampled_effects


class AudioTransformsExclusive:
    def __init__(self):
        self.probas = []
        self.transforms = []

    def add(self, proba, transform):
        self.probas.append(proba)
        self.transforms.append(transform)
        return self

    def sample(self):
        probas = np.array(self.probas) / np.sum(self.probas)
        selected_idx = np.random.choice(len(probas), size=1, p=probas)[0]
        return self.transforms[selected_idx].sample()


class _AbstractAudioTransform:
    _name = None

    def __init__(self, proba, *options):
        self.proba = proba
        self.options = options

    def sample(self):
        if np.random.uniform(0.0, 1.0) < self.proba:
            effect = []
            for option in self.options:
                if isinstance(option, list):
                    low, high, *option = option
                    magnitude = np.random.uniform(low=low, high=high)
                    magnitude = magnitude if len(option) == 0 else np.power(10, magnitude)
                else:
                    magnitude = option
                if isinstance(magnitude, float):
                    effect.append('{0:.4f}'.format(magnitude))
                else:
                    effect.append(str(magnitude))
            return [[self._name] + effect]


def make_transform(name):
    class _AudioTransform(_AbstractAudioTransform):
        _name = name

    class_name = name.title() + 'AudioTransform'
    _AudioTransform.__name__ = class_name
    _AudioTransform.__qualname__ = class_name
    return _AudioTransform


def get_default_audio_transforms():
    return AudioTransformsChain(
    ).add_group(
        0.2, [
            AudioTransformsExclusive(
            ).add(
                0.5,
                # The frequency response drops logarithmically around the center frequency. band center width
                make_transform('band')(1.0, [2, 3.8, 'log'], [2.5, 3.8, 'log'])
            ).add(
                0.5,
                # Non linear distortion.
                make_transform('overdrive')(1.0, [0, 10], 20)
            )
        ]
    ).add_group(
        0.3, [
            AudioTransformsExclusive(
            ).add(
                1.0,
                # Change the audio pitch (but not tempo).
                make_transform('pitch')(1.0, [-100, 100]),
            ).add(
                # Note: I prefer to disable this transformation because audio squeezing may produce NaNs in CTC loss
                #     and I rather to disable it than introduce bias
                0.0,
                # Adjust the audio speed (pitch and tempo together)
                make_transform('speed')(1.0, [0.9, 1.1]),
            ).add(
                # Note: I prefer to disable this transformation because audio squeezing may produce NaNs in CTC loss
                #     and I rather to disable it than introduce bias
                0.0,
                # Change the audio playback speed but not its pitch.
                make_transform('tempo')(1.0, [0.8, 1.2])
            )
        ]
    ).add_group(
        0.3, [
            AudioTransformsExclusive(
            ).add(
                0.5,
                # Boost or cut the bass (lower) frequencies. bass gain frequency width
                make_transform('bass')(1.0, [-20, 20]),
            ).add(
                0.5,
                # Boost or cut the treble (upper) frequencies. treble gain frequency width
                make_transform('treble')(1.0, [-20, 20])
            )
        ]
    ).add_group(
        0.2, [
            AudioTransformsExclusive(
            ).add(
                0.5,
                # Add a chorus effect to the audio. gain-in gain-out <delay decay speed depth −s|−t>
                make_transform('chorus')(1.0, [0.5, 1.0], [0.5, 1.0], [40, 60], 0.4, 0.25, 2, '-t'),
            ).add(
                0.5,
                # Add reverberation to the audio
                make_transform('reverb')(1.0)
            )
        ]
    ).add_group(
        0.3, [
            # DC shift to the audio
            make_transform('dcshift')(1.0, [-0.2, 0.2]),

            # Delay one or more audio channels such that they start at the given position.
            make_transform('delay')(1.0, [0.0, 1.0])
        ]
    )


class SpectrogramTransform(torch.nn.Module):
    def __init__(self, freq_mask_param=10, time_mask_param=10):
        super().__init__()

        self.freq_mask_param = freq_mask_param
        self.time_mask_param = time_mask_param

        self._time_masking = torchaudio.transforms.TimeMasking(self.time_mask_param)
        self._frequency_masking = torchaudio.transforms.FrequencyMasking(self.freq_mask_param)

        self._half_time_masking = torchaudio.transforms.TimeMasking(self.time_mask_param // 2)
        self._half_frequency_masking = torchaudio.transforms.FrequencyMasking(self.freq_mask_param // 2)

    def forward(self, spectrogram):
        t_type = np.random.choice(['_frequency_masking', '_time_masking', '_both', '_none'])

        if t_type == '_frequency_masking':
            return self._frequency_masking(spectrogram)
        elif t_type == '_time_masking':
            return self._time_masking(spectrogram)
        elif t_type == '_both':
            return self._half_time_masking(self._half_frequency_masking(spectrogram))
        else:
            return spectrogram


def compute_log_mel_spectrogram(
        audio, sequence_lengths,
        sample_rate=8000, window_size=0.02, window_step=0.01,
        f_min=20, f_max=3800, n_mels=64, window_fn=torch.hamming_window,
        power=1.0, eps=1e-6, spectrogram_transform=None
):
    """ Compute log-mel spectrogram.
        Input shape:
            audio: 3D tensor with shape (batch_size, num_timesteps)
            sequence_lengths: 1D tensor with shape (batch_size)
        Returns:
            4D tensor with shape (batch_size, n_mels, new_num_timesteps)
            1D tensor with shape (batch_size)
    """
    spectrogram_transform = (lambda x: x) if spectrogram_transform is None else spectrogram_transform

    win_length = int(window_size * sample_rate)
    hop_length = int(window_step * sample_rate)

    # write your code here
    mel_transformer = torchaudio.transforms.MelSpectrogram(
        sample_rate=sample_rate, win_length=win_length, hop_length=hop_length, n_fft=win_length,
        f_min=f_min, f_max=f_max, pad=0, n_mels=n_mels, power=power, window_fn=window_fn
    ).to(device=audio.device)

    log_mel_spectrogram = torch.log(mel_transformer(audio) + eps)
    log_mel_spectrogram = spectrogram_transform(log_mel_spectrogram)
    seq_len = ((sequence_lengths + 2 * hop_length - win_length) // hop_length + 1).to(dtype=torch.long)

    return log_mel_spectrogram, seq_len
