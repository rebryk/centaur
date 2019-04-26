from typing import Tuple

import librosa
import numpy as np
from scipy import signal


# TODO: add normalization
class SoundConverter:
    def __init__(self,
                 sr: int = 22050,
                 n_fft: int = 1024,
                 power: float = 1.0,
                 n_mel: int = 80,
                 n_mag: int = 513,
                 fmin: float = 1e-5,
                 fmax: float = None,
                 htk: bool = True,
                 norm: int = None,
                 trim: bool = False,
                 preemphasis: float = None):
        self.sr = sr
        self.n_fft = n_fft
        self.power = power
        self.n_mel = n_mel
        self.n_mag = n_mag
        self.trim = trim
        self.preemphasis = preemphasis

        self._mel_basis = librosa.filters.mel(
            sr=self.sr,
            n_fft=self.n_fft,
            n_mels=self.n_mel,
            fmin=fmin,
            fmax=fmax,
            htk=htk,
            norm=norm
        )

    def read_wav(self, path: str) -> np.array:
        wav, _ = librosa.core.load(path, sr=self.sr)
        return wav

    def wav_to_mag(self, wav: np.array) -> np.array:
        if self.trim:
            wav, _ = librosa.effects.trim(wav)

        if self.preemphasis:
            wav = np.append(wav[0], wav[1:] - self.preemphasis * wav[:-1])

        complex_spec = librosa.stft(y=wav, n_fft=self.n_fft)
        mag, _ = librosa.magphase(complex_spec, power=self.power)
        return mag.T

    def mag_to_mel(self, mag: np.array) -> np.array:
        mel = np.dot(self._mel_basis, mag.T)
        mel = np.log(np.clip(mel, a_min=1e-5, a_max=None)).T
        return mel

    def mel_to_mag(self, mel: np.array) -> np.array:
        mel = np.exp(mel)
        mag = np.dot(mel, self._mel_basis)
        mag = np.power(mag, 1. / self.power)
        return mag

    def mel_to_wav(self, mel: np.array, power: float = 1.5) -> np.array:
        mag = self.mel_to_mag(mel)
        wav = self.mag_to_wav(mag, power=power)
        return wav

    def mag_to_wav(self, mag: np.array, power: float = 1.5) -> np.array:
        mag = np.clip(mag, a_min=0, a_max=255)

        wav = self.griffin_lim(mag=mag.T ** power, n_iters=50, n_fft=self.n_fft)
        wav /= np.max(np.abs(wav))

        if self.preemphasis:
            wav = signal.lfilter([1], [1, - self.preemphasis], wav)

        if self.trim:
            wav, _ = librosa.effects.trim(wav)

        return wav

    def get_audio_features(self, wav_name: str) -> Tuple[np.array, np.array]:
        wav = self.read_wav(wav_name)
        mag = self.wav_to_mag(wav)
        mel = self.mag_to_mel(mag)
        return mel, mag

    @staticmethod
    def griffin_lim(mag: np.array, n_iters: int, n_fft: int) -> np.array:
        phase = np.exp(2j * np.pi * np.random.rand(*mag.shape))
        complex_spec = mag * phase
        wav = librosa.istft(complex_spec)

        if not np.isfinite(wav).all():
            print('WARNING: audio was not finite, skipping audio saving')
            return np.array([0])

        for _ in range(n_iters):
            _, phase = librosa.magphase(librosa.stft(wav, n_fft=n_fft))
            complex_spec = mag * phase
            wav = librosa.istft(complex_spec)

        return wav
