import os

import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torch.utils.data.distributed import DistributedSampler

import utils
from sound import SoundConverter
from text import text_to_sequence


class LJDataset(Dataset):
    """LJSpeech dataset."""

    def __init__(self,
                 csv_file: str,
                 root_dir: str,
                 sound_converter: SoundConverter,
                 cleaners: str = 'english_cleaners',
                 use_cache: bool = True,
                 sort_by_len: bool = False,
                 n_samples: int = None):
        """
        :param csv_file: path to the csv file with annotations
        :param root_dir: directory with all the wavs
        :param cleaners:
        :param sound_converter: the sound converter
        :param use_cache: whether to use cache
        :param sort_by_len: whether to sort the dataset by text length
        :param n_samples: number of samples to use
        """

        self.landmarks_frame = pd.read_csv(csv_file, sep='|', header=None)

        if sort_by_len:
            self.landmarks_frame = self.sort_dataset(self.landmarks_frame)

        self.root_dir = root_dir
        self.cleaners = cleaners
        self.sound_converter = sound_converter
        self.use_cache = use_cache
        self._cache = {}
        self.n_samples = n_samples

    def __len__(self) -> int:
        length = len(self.landmarks_frame)
        n_samples = self.n_samples or length
        return min(length, n_samples)

    def __getitem__(self, idx: int) -> dict:
        if self.use_cache and idx in self._cache:
            return self._cache[idx]

        text = self.landmarks_frame.ix[idx, 1]
        text = np.asarray(text_to_sequence(text, [self.cleaners]), dtype=np.int32)

        wav_name = os.path.join(self.root_dir, self.landmarks_frame.ix[idx, 0]) + '.wav'

        mel, mag = self.sound_converter.get_audio_features(wav_name)
        spec = np.concatenate([mel, mag], axis=-1)
        sample = {'text': text, 'spec': spec}

        if self.use_cache:
            self._cache[idx] = sample

        return sample

    @staticmethod
    def sort_dataset(df: pd.DataFrame) -> pd.DataFrame:
        index = df[2].str.len().sort_values().index
        df = df.reindex(index).reset_index(drop=True)
        return df


def get_loader(data_path: str,
               file_name: str,
               sound_converter: SoundConverter,
               batch_size: int,
               is_distributed: bool = False,
               n_workers: int = 4,
               sort_by_len: bool = False,
               n_samples: int = None) -> DataLoader:
    dataset = LJDataset(
        csv_file=os.path.join(data_path, file_name),
        root_dir=os.path.join(data_path, 'wavs'),
        sound_converter=sound_converter,
        sort_by_len=sort_by_len,
        n_samples=n_samples
    )

    sampler = DistributedSampler(dataset) if is_distributed else None

    # TODO: shuffle?
    data_loader = DataLoader(
        dataset=dataset,
        sampler=sampler,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=utils.collate_fn,
        drop_last=True,
        pin_memory=False,
        num_workers=n_workers
    )

    return data_loader
