import argparse
import time
import os
from typing import Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import utils
from dataset import get_loader
from distributed import init_distributed, apply_gradient_allreduce, reduce_tensor
from distributed import train_distributed
from logger import Logger
from loss import Loss
from model import Model
from optimizer import Novograd
from scheduler import PolyLR
from sound import SoundConverter
from text import n_symbols


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, required=True,
                        help='path to configuration file')
    parser.add_argument('-d', '--distributed', action='store_true',
                        help='enable distributed learning')
    parser.add_argument('-r', '--rank', type=int, default=0,
                        help='rank of process for distributed')
    parser.add_argument('-g', '--group_name', type=str, default='',
                        help='name of group for distributed')
    return parser.parse_args()


def validation(model: Model,
               loss: Loss,
               loader: DataLoader) -> Tuple[dict, dict, torch.Tensor]:
    model = utils.clone_model(model).cuda()
    model.eval()

    valid_loss = 0

    with torch.no_grad():
        for i, batch in enumerate(loader):
            batch = utils.to_cuda(batch)
            model_output = model.infer(
                text=batch['text'],
                text_mask=batch['text_mask'],
                n_features=593,
                max_length=batch['max_length']
            )

            loss_value = loss(
                mel_output=model_output['mel'],
                mel_target=batch['target'][:, :, :80],
                mag_output=model_output['mag'],
                mag_target=batch['target'][:, :, 80:],
                stop_token_output=model_output['stop_token_logits'],
                stop_token_target=batch['stop_token'],
                length=batch['length']
            )

            valid_loss = valid_loss * i / (i + 1) + loss_value / (i + 1)

    return batch, model_output, valid_loss


def training_step(model: Model,
                  loss: Loss,
                  optimizer: nn.Module,
                  batch: dict,
                  n_gpus: int,
                  clip_grad_norm: float = None) -> Tuple[dict, torch.Tensor]:
    model.train()

    model.zero_grad()

    batch = utils.to_cuda(batch)

    model_output = model(
        text=batch['text'],
        text_mask=batch['text_mask'],
        spec=batch['input'],
        max_length=batch['max_length']
    )

    loss_value = loss(
        mel_output=model_output['mel'],
        mel_target=batch['target'][:, :, :80],
        mag_output=model_output['mag'],
        mag_target=batch['target'][:, :, 80:],
        stop_token_output=model_output['stop_token_logits'],
        stop_token_target=batch['stop_token'],
        length=batch['length']
    )

    # TODO: add float16
    loss_value.backward()

    if clip_grad_norm is not None:
        nn.utils.clip_grad_norm_(model.parameters(), clip_grad_norm)

    optimizer.step()

    if n_gpus > 1:
        loss_value = reduce_tensor(loss_value.data, n_gpus)

    return model_output, loss_value


def train(n_gpus: int, config_path: str, rank: int, group_name: str):
    config = utils.read_config(config_path)

    # Fix random seeds
    utils.fix_seed(rank)

    if n_gpus > 1:
        # TODO: remove backend and url
        init_distributed(n_gpus, rank, group_name, 'nccl', 'tcp://localhost:54321')

    loss = Loss()
    model = Model(
        n_symbols=n_symbols,
        encoder_hidden_size=config.model.encoder_hidden_size,
        decoder_hidden_size=config.model.decoder_hidden_size,
        reduction_factor=config.model.reduction_factor
    )
    model = model.cuda()

    if n_gpus > 1:
        model = apply_gradient_allreduce(model)

    optimizer = Novograd(
        params=model.parameters(),
        lr=config.training.lr,
        weight_decay=config.training.weight_decay
    )
    scheduler = PolyLR(optimizer, 1e-5, config.training.epochs)

    global_step = 0

    if config.training.checkpoint_path != '':
        model, optimizer, scheduler, global_step = utils.load_checkpoint(
            checkpoint_path=config.training.checkpoint_path,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler
        )

    # TODO: add float16
    # TODO: load checkpoint

    sound_converter = SoundConverter()
    train = get_loader(
        data_path=config.data.path,
        file_name=config.data.train,
        sound_converter=sound_converter,
        batch_size=config.training.batch_size,
        is_distributed=n_gpus > 1
    )
    valid = get_loader(
        data_path=config.data.path,
        file_name=config.data.valid,
        sound_converter=sound_converter,
        batch_size=config.training.batch_size,
        sort_by_len=True,
        n_samples=config.data.get('n_valid_samples', None)
    )

    logger = Logger(config.training.log_path, sound_converter) if rank == 0 else None
    epoch_offset = max(0, int(global_step / len(train)))
    start_time = time.time()

    for epoch in range(epoch_offset, config.training.epochs):
        for i, batch in enumerate(train):
            global_step += 1
            scheduler.step()

            train_output, train_loss = training_step(
                model=model,
                loss=loss,
                optimizer=optimizer,
                batch=batch,
                n_gpus=n_gpus,
                clip_grad_norm=config.training.get('clip_grad_norm', None)
            )

            if global_step % config.training.log_step == 0 and rank == 0:
                valid_batch, valid_output, valid_loss = validation(model, loss, valid)

                print(f'Step: {global_step}')
                print(f'Train loss: {train_loss:0.4f}')
                print(f'Valid loss: {valid_loss:0.4f}')

                logger.add_scalar('train_loss', train_loss, global_step)
                logger.add_sample('train', batch, train_output, 0, global_step)

                logger.add_scalar('valid_loss', valid_loss, global_step)
                logger.add_sample('valid', valid_batch, valid_output, 0, global_step)

                end_time = time.time()
                print(f'Time per step: {(end_time - start_time) / config.training.log_step:0.2f}s')
                start_time = end_time

            if global_step % config.training.save_step == 0 and rank == 0:
                checkpoint_path = f'{config.training.output_path}/checkpoint_{global_step}'
                utils.save_checkpoint(model, optimizer, scheduler, global_step, checkpoint_path)


if __name__ == '__main__':
    args = parse_args()

    config = utils.read_config(args.config)
    output_path = config.training.output_path

    if not os.path.isdir(output_path):
        os.makedirs(output_path)
        os.chmod(output_path, 0o775)

    if args.distributed:
        train_distributed(args.config, output_path)
    else:
        n_gpus = torch.cuda.device_count()

        if n_gpus > 1:
            if args.group_name == '':
                print('WARNING: Multiple GPUs detected but no distributed group set')
                print('Only running 1 GPU.  Use distributed.py for multiple GPUs')
                n_gpus = 1

        if n_gpus == 1 and args.rank != 0:
            raise Exception('Doing single GPU training on rank > 0')

        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = False
        train(n_gpus, args.config, args.rank, args.group_name)
