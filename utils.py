import json
import os
import random
from typing import Any

import numpy as np
import torch
from attrdict import AttrDict
from torch.optim.lr_scheduler import _LRScheduler

from model import Model
from text import n_symbols


def _get_padded_length(length: int, pad_to: int = 8) -> int:
    return length + pad_to - (length % pad_to)


def pad(x: np.array, length: int, value: float = 0.0) -> np.array:
    n, _ = x.shape[0], x.shape[1:]
    shape = (length - n,) + _
    padding = np.zeros(shape, dtype=np.float32) + value
    return np.concatenate([x, padding], axis=0)


def to_cuda(x: dict) -> dict:
    return {key: value.cuda() for key, value in x.items()}


def fix_seed(seed: int = 0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def to_numpy(x: Any) -> Any:
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()

    if isinstance(x, dict):
        return {key: to_numpy(value) for key, value in x.items()}

    if isinstance(x, list):
        return [to_numpy(value) for value in x]

    return x


def load_checkpoint(checkpoint_path: str,
                    model: Model,
                    optimizer: torch.optim.Optimizer,
                    scheduler: _LRScheduler):
    assert os.path.isfile(checkpoint_path)

    checkpoint_dict = torch.load(checkpoint_path, map_location='cpu')
    global_step = checkpoint_dict['global_step']
    optimizer.load_state_dict(checkpoint_dict['optimizer'])
    scheduler.load_state_dict(checkpoint_dict['scheduler'])
    model_for_loading = checkpoint_dict['model']
    model.load_state_dict(model_for_loading.state_dict())
    return model, optimizer, scheduler, global_step


def read_config(config_path: str) -> AttrDict:
    with open(config_path, 'r') as config_file:
        return AttrDict(json.load(config_file))


def clone_model(model: Model) -> Model:
    state = model.state_dict()
    model = Model(
        n_symbols=n_symbols,
        encoder_hidden_size=model.encoder_hidden_size,
        decoder_hidden_size=model.decoder_hidden_size,
        reduction_factor=model.reduction_factor
    )
    model.load_state_dict(state)

    return model


def save_checkpoint(model: Model,
                    optimizer: torch.optim.Optimizer,
                    scheduler: _LRScheduler,
                    global_step: int,
                    filepath: str):
    print(f'Saving model and optimizer state at iteration {global_step} to {filepath}')
    model_for_saving = clone_model(model).cuda()

    obj = {
        'model': model_for_saving,
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict(),
        'global_step': global_step
    }
    torch.save(obj, filepath)


def collate_fn(batch: dict) -> dict:
    batch_size = len(batch)

    text = [it['text'] for it in batch]
    text_length = np.array([len(it) for it in text], dtype=np.int32)
    max_text_length = _get_padded_length(np.max(text_length))

    text_mask = np.zeros([batch_size, max_text_length], dtype=np.int32)
    for i, it in enumerate(text_length):
        text_mask[i, :it] = 1

    text_padded = np.zeros([batch_size, max_text_length], dtype=np.int32)
    for i, it in enumerate(text):
        text_padded[i, :len(it)] = it

    spec = [it['spec'] for it in batch]
    spec_length = np.array([len(it) for it in spec], dtype=np.int32)
    max_spec_length = _get_padded_length(np.max(spec_length))

    n_features = spec[0].shape[-1]
    spec_padded = np.zeros([batch_size, max_spec_length, n_features], dtype=np.float32)
    for i, it in enumerate(spec):
        spec_padded[i, :len(it), :] = it

    spec_input = np.concatenate(
        [np.zeros([batch_size, 1, n_features], np.float32), spec_padded[:, :-1, :]],
        axis=1
    )

    stop_token = np.zeros([batch_size, max_spec_length], dtype=np.int32)
    for i, it in enumerate(spec_length):
        stop_token[i, it - 1:] = 1

    output = {
        'text': torch.tensor(text_padded, dtype=torch.int64),
        'text_mask': torch.tensor(text_mask, dtype=torch.int32),
        'target': torch.tensor(spec_padded, dtype=torch.float32),
        'input': torch.tensor(spec_input, dtype=torch.float32),
        'length': torch.tensor(spec_length, dtype=torch.int32),
        'max_length': torch.tensor(max_spec_length, dtype=torch.int32),
        'stop_token': torch.tensor(stop_token, dtype=torch.float32)
    }

    return output
