import torch
import torch.nn as nn
import torch.nn.functional as F


class Loss(nn.Module):
    def __init__(self,
                 mel_weight: float = 1.0,
                 mag_weight: float = 1.0,
                 stop_token_weight: float = 1.0):
        super(Loss, self).__init__()

        self.mel_weight = mel_weight
        self.mag_weight = mag_weight
        self.stop_token_weight = stop_token_weight

    def __call__(self,
                 mel_output: torch.Tensor,
                 mel_target: torch.Tensor,
                 mag_output: torch.Tensor,
                 mag_target: torch.Tensor,
                 stop_token_output: torch.Tensor,
                 stop_token_target: torch.Tensor,
                 length: torch.Tensor):
        mel_mask = torch.zeros_like(mel_output, dtype=torch.float32)
        mag_mask = torch.zeros_like(mag_output, dtype=torch.float32)
        stop_token_mask = torch.zeros_like(stop_token_output, dtype=torch.float32)

        for i, it in enumerate(length):
            mel_mask[i, :it, :] = 1
            mag_mask[i, :it, :] = 1
            stop_token_mask[i, :it] = 1

        mel_loss = self.mel_weight * torch.mean(torch.abs(mel_output - mel_target) * mel_mask)
        mag_loss = self.mag_weight * torch.mean(torch.abs(mag_output - mag_target) * mag_mask)

        stop_token_loss = F.binary_cross_entropy_with_logits(
            input=stop_token_output,
            target=stop_token_target,
            reduction='none'
        )
        stop_token_loss = self.stop_token_weight * torch.mean(stop_token_loss * stop_token_mask)

        return mel_loss + mag_loss + stop_token_loss
