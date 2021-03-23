import os
import json

import regex
import numpy as np
import pandas as pd

import torch
import torch.utils.data
from torch.utils.data import DataLoader, BatchSampler, SequentialSampler, RandomSampler

from src.audio_utils import open_audio


class AudioDataset(torch.utils.data.Dataset):
    @staticmethod
    def load_constraint_dataset(path, min_duration, max_duration):
        min_duration = 0.0 if min_duration is None else min_duration
        max_duration = 1e10 if max_duration is None else max_duration
        print(path)
        print(min_duration, max_duration)
        dataset = pd.read_csv(path, header=None, names=['audio_path', 'text', 'duration'])
        dataset['duration'] = dataset['duration'].astype(float)
        # print(dataset)
        dataset = dataset[dataset['duration'] > min_duration]
        # print(dataset)
        dataset = dataset[dataset['duration'] < max_duration]
        # print(dataset)
        return dataset

    def __init__(
            self, dataset_path, vocab, sample_rate=8000,
            audio_transforms=None, min_duration=None, max_duration=None, evaluate_stats=False
    ):
        self._epoch = 0

        self.vocab = vocab
        self.sample_rate = sample_rate
        self.min_duration = min_duration
        self.max_duration = max_duration
        self.audio_transforms = audio_transforms

        if isinstance(dataset_path, list):
            data = pd.DataFrame()
            self.min_duration = (
                 self.min_duration if isinstance(self.min_duration, list) else [self.min_duration] * len(dataset_path)
            )
            self.max_duration = (
                self.max_duration if isinstance(self.max_duration, list) else [self.max_duration] * len(dataset_path)
            )
            for min_duration, max_duration, path in zip(self.min_duration, self.max_duration, dataset_path):
                dataset = self.load_constraint_dataset(path, min_duration, max_duration)
                data = data.append(dataset)
        else:
            data = self.load_constraint_dataset(dataset_path, self.min_duration, self.max_duration)

        self.data = data.sort_values(by='duration')

        self.idx_to_text_len = dict()
        self.idx_to_audio_len = dict()
        if evaluate_stats:
            for idx in range(self.data.shape[0]):
                self.idx_to_text_len[idx] = len(self.data.iloc[idx]['text'])
                self.idx_to_audio_len[idx] = self.data.iloc[idx]['duration']

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx, supress_effects=False):
        apply_transforms = (self.audio_transforms is not None) and (not supress_effects)

        text = self.data.iloc[idx]['text']
        audio_path = self.data.iloc[idx]['audio_path']

        # write your code here
        text_len = len(text)
        tokens = torch.tensor(self.vocab.lookup_indices(text))
        audio, audio_len = open_audio(
            audio_path, self.sample_rate,
            effects=self.audio_transforms.sample() if apply_transforms else None
        )

        return {"audio": audio,  # torch tensor, (num_timesteps)
                "audio_len": audio_len,  # int
                "text": text,  # str
                "text_len": text_len,  # int
                'tokens': tokens,  # torch tensor, (text_len)
                }


def convert_libri_manifest_to_common_voice(manifest_path):
    cv_manifest_path = manifest_path.replace('.json', '.common_voice.csv')
    with open(manifest_path, 'r') as in_file:
        with open(cv_manifest_path, 'w') as out_file:
            for line in in_file:
                sample = json.loads(line, parse_float=lambda x: x)
                audio_filepath = os.path.join(
                    os.path.dirname(os.path.abspath(manifest_path)), sample['audio_filepath']
                )
                out_file.write(','.join([audio_filepath, sample['text'], sample['duration']]) + '\n')

    return cv_manifest_path


def convert_open_stt_manifest_to_common_voice(manifest_path, min_duration=2.0):
    cv_manifest_path = manifest_path.replace('.csv', '.common_voice.csv')

    with open(manifest_path, 'r') as in_file:
        with open(cv_manifest_path, 'w') as out_file:
            for line in in_file:
                audio_filepath, test_filepath, duration = line.strip().split(',')
                if float(duration) < min_duration:
                    continue

                audio_filepath = os.path.join(
                    os.path.dirname(os.path.abspath(manifest_path)), './..', audio_filepath
                )
                test_filepath = os.path.join(
                    os.path.dirname(os.path.abspath(manifest_path)), './..', test_filepath
                )
                text = ' '.join(map(str.strip, open(test_filepath, 'r').readlines()))
                text = regex.sub(r'\P{Cyrillic}', ' ', text)
                text = regex.sub(' +', ' ', text)

                out_file.write(','.join([audio_filepath, text, duration]) + '\n')
    return cv_manifest_path


def manifest_train_test_split(manifest_path, ratio=0.3):
    test_manifest_path = manifest_path.replace('.csv', '_test.csv')
    train_manifest_path = manifest_path.replace('.csv', '_train.csv')

    data = pd.read_csv(manifest_path)
    permutation = np.random.permutation(data.shape[0])
    data = data.iloc[permutation]

    test_size = int(ratio * data.shape[0])
    test_data = data[:test_size]
    train_data = data[test_size:]
    test_data.to_csv(test_manifest_path, index=False)
    train_data.to_csv(train_manifest_path, index=False)

    return test_manifest_path, train_manifest_path


def collate_fn(batch, get_batch_memory_usage=True):
    """
        Inputs:
            batch: list of elements with length=batch_ize
        Returns:
            dict
    """
    # write your code here
    audios = torch.nn.utils.rnn.pad_sequence(
        [obj['audio'] for obj in batch], batch_first=True, padding_value=0.0
    )
    audio_lens = torch.tensor([obj['audio'].shape[0] for obj in batch])

    texts = [obj['text'] for obj in batch]
    text_lens = torch.tensor([len(obj['text']) for obj in batch])
    tokens = torch.nn.utils.rnn.pad_sequence(
        [obj['tokens'] for obj in batch], batch_first=True, padding_value=0.0
    )
    batch = {
        'audios': audios,  # torch tensor, (batch_size, max_num_timesteps)
        'audio_lens': audio_lens,  # torch tensor, (batch_size)
        'texts': texts,  # list, len=(batch_size)
        'text_lens': text_lens,  # torch tensor, (batch_size)
        'tokens': tokens,  # torch tensor, (batch_size, max_text_len)
    }

    if get_batch_memory_usage:
        batch_memory_usage = 0.0
        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                batch_memory_usage += value.numel()
        batch['memory_usage'] = batch_memory_usage

    return batch


class AudioDatasetSampler:
    def __init__(self, dataset, batch_size):
        self.epoch = 0
        self.dataset = dataset
        self.batch_size = batch_size

        # Assume that data in dataset is sorted w.r. to duration
        self.batches = list(BatchSampler(SequentialSampler(self.dataset), batch_size=self.batch_size, drop_last=False))

    def __iter__(self):
        if self.epoch == 0:
            for batch_idx in SequentialSampler(self.batches):
                for idx in self.batches[batch_idx]:
                    yield idx
        else:
            for batch_idx in RandomSampler(self.batches):
                for idx in self.batches[batch_idx]:
                    yield idx
        self.epoch += 1

    def __len__(self):
        return len(self.dataset)
