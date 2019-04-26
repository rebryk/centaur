from typing import List, Any

import matplotlib.pyplot as plt
import numpy as np
from tensorboardX import SummaryWriter

import utils
from sound import SoundConverter


class Logger:
    def __init__(self,
                 log_dir: str,
                 sound_converter: SoundConverter = None):
        self.sound_converter = sound_converter
        self._summary_writer = SummaryWriter(log_dir=log_dir)

    def add_spectrograms(self,
                         tag: str,
                         specs: List[np.array],
                         titles: List[str],
                         stop_token_output: np.array,
                         stop_token_target: np.array,
                         audio_length: int,
                         global_step: int):
        num_figs = len(specs) + 1
        fig, ax = plt.subplots(nrows=num_figs, figsize=(8, num_figs * 3))

        for i, (spec, title) in enumerate(zip(specs, titles)):
            spec = np.pad(spec, ((1, 1), (1, 1)), 'constant', constant_values=0.)
            spec = spec.astype(float)
            colour = ax[i].imshow(spec.T, cmap='viridis', interpolation=None, aspect='auto')
            ax[i].invert_yaxis()
            ax[i].set_title(title)
            fig.colorbar(colour, ax=ax[i])

        stop_token_target = stop_token_target.astype(float)
        ax[-1].plot(stop_token_target, 'r.')

        stop_token_output = stop_token_output.astype(float)
        ax[-1].plot(stop_token_output, 'g.')
        ax[-1].axvline(x=audio_length)
        ax[-1].set_xlim(0, len(specs[0]))
        ax[-1].set_title('stop token')

        plt.xlabel('time')
        plt.tight_layout()

        cb = fig.colorbar(colour, ax=ax[-1])
        cb.remove()

        self._summary_writer.add_figure(tag, fig, global_step)
        plt.close(fig)

    def add_mel(self, tag: str, mel: np.array, global_step: int):
        wav = self.sound_converter.mel_to_wav(mel)
        self._summary_writer.add_audio(tag, wav, global_step, self.sound_converter.sr)

    def add_mag(self, tag: str, mag: np.array, global_step: int):
        wav = self.sound_converter.mag_to_wav(mag)
        self._summary_writer.add_audio(tag, wav, global_step, self.sound_converter.sr)

    def add_scalar(self, tag: str, value: Any, global_step: int):
        self._summary_writer.add_scalar(tag, value, global_step)

    def add_summary(self,
                    tag: str,
                    mel_output: np.array,
                    mel_target: np.array,
                    mag_output: np.array,
                    mag_target: np.array,
                    stop_token_output: np.array,
                    stop_token_target: np.array,
                    attentions: List[np.array],
                    audio_length: int,
                    global_step: int):
        max_length = max(len(mel_output), len(mel_target))

        # Add padding
        mel_output = utils.pad(mel_output, max_length, 0.0)
        mel_target = utils.pad(mel_target, max_length, 0.0)
        mag_output = utils.pad(mag_output, max_length, 0.0)
        mag_target = utils.pad(mag_target, max_length, 0.0)
        stop_token_output = utils.pad(stop_token_output, max_length, 0.0)
        stop_token_target = utils.pad(stop_token_target, max_length, 1.0)

        titles = ['mel target', 'mel output']
        specs = [mel_target, mel_output]

        for layer, values in enumerate(attentions):
            for head, value in enumerate(values):
                titles.append(f'encoder-decoder attention layer {layer}, head {head}')
                specs.append(value)

        titles += ['mag target', 'mag output']
        specs += [mag_target, mag_output]

        self.add_spectrograms(
            tag=f'{tag}_image',
            specs=specs,
            titles=titles,
            stop_token_output=stop_token_output,
            stop_token_target=stop_token_target,
            audio_length=audio_length,
            global_step=global_step
        )

        self.add_mel(f'{tag}_mel_output', mel_output, global_step)
        self.add_mag(f'{tag}_mag_output', mag_output, global_step)
        # self.add_mel(f'{tag}_mel_target', mel_target[:audio_length], global_step)
        # self.add_mag(f'{tag}_mag_target', mag_target[:audio_length], global_step)

    def add_sample(self,
                   tag: str,
                   batch: dict,
                   model_output: dict,
                   index: int,
                   global_step: int):
        # Convert tensors to numpy
        batch = utils.to_numpy(batch)
        model_output = utils.to_numpy(model_output)

        self.add_summary(
            tag=tag,
            mel_output=model_output['mel'][index],
            mel_target=batch['target'][index][:, :80],
            mag_output=model_output['mag'][index],
            mag_target=batch['target'][index][:, 80:],
            stop_token_output=model_output['stop_token'][index],
            stop_token_target=batch['stop_token'][index],
            attentions=[it[index] for it in model_output['attentions']],
            audio_length=batch['length'][index],
            global_step=global_step
        )
