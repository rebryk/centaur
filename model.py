import copy
from typing import Callable

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


def clones(module: nn.Module, n: int) -> nn.ModuleList:
    return nn.ModuleList([copy.deepcopy(module) for _ in range(n)])


class Linear(nn.Module):
    def __init__(self,
                 in_features: int,
                 out_features: int,
                 bias: bool = True):
        super(Linear, self).__init__()
        self.linear = nn.Linear(in_features, out_features, bias)
        nn.init.xavier_uniform_(self.linear.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


class FeedForwardNetwork(nn.Module):
    def __init__(self, input_size: int, output_size: int, hidden_size: int, dropout: float):
        super(FeedForwardNetwork, self).__init__()
        self.linear_1 = Linear(input_size, hidden_size)
        self.linear_2 = Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear_2(self.dropout(F.relu(self.linear_1(x))))


class DecoderPrenet(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, output_size: int, dropout: float):
        super(DecoderPrenet, self).__init__()
        self.linear_1 = Linear(input_size, hidden_size)
        self.linear_2 = Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.dropout(F.relu(self.linear_1(x)))
        x = self.dropout(F.relu(self.linear_2(x)))
        return x


class SublayerConnection(nn.Module):
    def __init__(self, hidden_size: int, dropout: float):
        super(SublayerConnection, self).__init__()
        self.layer_norm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, sublayer: Callable) -> torch.Tensor:
        return x + self.dropout(sublayer(self.layer_norm(x)))


class ConvBlock(nn.Module):
    def __init__(self,
                 input_size: int,
                 output_size: int,
                 kernel_size: int = 3,
                 stride: int = 1,
                 dropout: float = 0.1,
                 is_causal: bool = False,
                 is_residual: bool = True,
                 activation: Callable = F.relu):
        super(ConvBlock, self).__init__()
        self.is_causal = is_causal
        self.is_residual = is_residual
        self.padding = int(kernel_size // 2)
        self.conv = nn.Conv1d(
            in_channels=input_size,
            out_channels=output_size,
            kernel_size=kernel_size,
            stride=stride
        )
        nn.init.xavier_uniform_(self.conv.weight)

        self.batch_norm = nn.BatchNorm1d(output_size)
        self.activation = activation
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = F.pad(x, [2 * self.padding, 0] if self.is_causal else [self.padding, self.padding])
        y = self.conv(y)
        y = self.batch_norm(y)
        y = self.activation(y)
        y = self.dropout(y)

        return x + y if self.is_residual else y


class Encoder(nn.Module):
    def __init__(self, n_symbols: int, hidden_size: int, n_layers: int = 4, dropout: float = 0.1):
        super(Encoder, self).__init__()
        self.embedding = nn.Embedding(n_symbols, hidden_size, padding_idx=0)
        conv_block = ConvBlock(input_size=hidden_size, output_size=hidden_size, dropout=dropout)
        self.layers = clones(conv_block, n_layers)
        # TODO: use bias?
        self.projection = Linear(hidden_size, hidden_size)
        self.pos_encoding = PositionalEncoding(hidden_size=hidden_size)

    def forward(self, text: torch.Tensor) -> torch.Tensor:
        x = self.embedding(text)
        x = x.transpose(1, 2)

        for layer in self.layers:
            x = layer(x)

        x = x.transpose(1, 2)
        x = self.projection(x)
        return self.pos_encoding(x)


class Attention(nn.Module):
    def __init__(self, dropout: float = 0.0):
        super(Attention, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.attention = None

    def forward(self,
                query: torch.Tensor,
                key: torch.Tensor,
                value: torch.Tensor,
                mask: torch.Tensor = None) -> torch.Tensor:
        key_size = query.size(-1)
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(key_size)

        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        self.attention = self.dropout(F.softmax(scores, dim=-1))

        return torch.matmul(self.attention, value)


class MultiHeadedAttention(nn.Module):
    def __init__(self, n_heads: int, hidden_size: int, dropout: float = 0.1):
        super(MultiHeadedAttention, self).__init__()

        assert hidden_size % n_heads == 0

        self.key_size = hidden_size // n_heads
        self.n_heads = n_heads
        self.linear_layers = clones(Linear(hidden_size, hidden_size), 4)
        self.attention = Attention(dropout)

    def forward(self,
                query: torch.Tensor,
                key: torch.Tensor,
                value: torch.Tensor,
                mask: torch.Tensor = None) -> torch.Tensor:
        if mask is not None:
            mask = mask.unsqueeze(1)

        batch_size = query.size(0)

        query, key, value = [
            layer(x).view(batch_size, -1, self.n_heads, self.key_size).transpose(1, 2)
            for layer, x in zip(self.linear_layers, (query, key, value))
        ]

        x = self.attention(query, key, value, mask=mask)
        x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.n_heads * self.key_size)

        return self.linear_layers[-1](x)


class PositionalEncoding(nn.Module):
    def __init__(self, hidden_size: int, dropout: float = 0.0, max_len: int = 4096, dtype=torch.float32):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(dropout)

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, hidden_size, dtype=dtype)
        position = torch.arange(0, max_len, dtype=dtype).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, hidden_size, 2, dtype=dtype) * -(math.log(10000.0) / hidden_size))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + Variable(self.pe[:, :x.size(1)], requires_grad=False)
        return self.dropout(x)


class DecoderLayer(nn.Module):
    def __init__(self,
                 hidden_size: int,
                 n_heads: int,
                 dropout: float):
        super(DecoderLayer, self).__init__()
        self.conv_block = ConvBlock(
            input_size=hidden_size,
            output_size=hidden_size,
            kernel_size=5,
            dropout=dropout,
            is_causal=True
        )
        self.attention = MultiHeadedAttention(n_heads=n_heads, hidden_size=hidden_size, dropout=dropout)
        self.feed_forward = FeedForwardNetwork(
            input_size=hidden_size,
            output_size=hidden_size,
            hidden_size=4 * hidden_size,
            dropout=0.0
        )
        self.sublayers = clones(SublayerConnection(hidden_size=hidden_size, dropout=dropout), 2)

    def forward(self, x: torch.Tensor, memory: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        x = x.transpose(1, 2)
        x = self.conv_block(x)
        x = x.transpose(1, 2)
        x = self.sublayers[0](x, lambda x: self.attention(x, memory, memory, mask))
        return self.sublayers[1](x, self.feed_forward)


class Decoder(nn.Module):
    def __init__(self,
                 n_mel: int,
                 n_mag: int,
                 hidden_size: int,
                 n_layers: int = 4,
                 n_heads: int = 1,
                 n_convs: int = 4,
                 n_mag_convs: int = 4,
                 prenet_dropout: float = 0.5,
                 dropout: float = 0.1,
                 reduction_factor: int = 1):
        super(Decoder, self).__init__()
        self.n_mel = n_mel
        self.n_mag = n_mag
        self.reduction_factor = reduction_factor
        self.prenet = DecoderPrenet(
            input_size=(n_mel + n_mag) * reduction_factor,
            hidden_size=hidden_size,
            output_size=hidden_size,
            dropout=prenet_dropout
        )
        self.projection = Linear(hidden_size, hidden_size)
        self.pos_encoding = PositionalEncoding(hidden_size=hidden_size)

        layer = DecoderLayer(hidden_size=hidden_size, n_heads=n_heads, dropout=dropout)
        self.layers = clones(layer, n_layers)
        self.layer_norm = nn.LayerNorm(hidden_size)

        conv_block = ConvBlock(
            input_size=hidden_size,
            output_size=hidden_size,
            kernel_size=5,
            dropout=dropout,
            is_causal=True
        )
        self.conv_layers = clones(conv_block, n_convs)
        self.mag_conv_layers = clones(conv_block, n_mag_convs)

        self.mel_projection = Linear(hidden_size, n_mel * reduction_factor)
        self.mag_projection = Linear(hidden_size, n_mag * reduction_factor)
        self.stop_token_projection = Linear(hidden_size, 1 * reduction_factor)

    def forward(self,
                x: torch.Tensor,
                memory: torch.Tensor,
                mask: torch.Tensor) -> [torch.Tensor, torch.Tensor, torch.Tensor, list]:
        # Shrink the input
        x = self._shrink(x, self.reduction_factor)

        # Pre-net and positional encoding
        x = self.prenet(x)
        x = self.pos_encoding(x)

        # Encoder-decoder attentions
        attentions = []

        for layer in self.layers:
            x = layer(x, memory, mask)
            attentions.append(layer.attention.attention.attention)

        # Final layer normalization
        x = self.layer_norm(x)

        # Apply convolution layers
        x = x.transpose(1, 2)

        for layer in self.conv_layers:
            x = layer(x)

        x = x.transpose(1, 2)

        # Predict stop token logits and mel-spectogram
        stop_token_logits = self.stop_token_projection(x)
        mel_spec = self.mel_projection(x)

        # Apply mag convolution layers
        x = x.transpose(1, 2)

        for layer in self.mag_conv_layers:
            x = layer(x)

        x = x.transpose(1, 2)

        mag_spec = self.mag_projection(x)

        # Expand the output
        mel_spec = self._expand(mel_spec, self.reduction_factor)
        mag_spec = self._expand(mag_spec, self.reduction_factor)
        stop_token_logits = self._expand(stop_token_logits, self.reduction_factor)

        return mel_spec, mag_spec, stop_token_logits.squeeze(dim=-1), attentions

    @staticmethod
    def _shrink(x: torch.Tensor, factor: int) -> torch.Tensor:
        batch_size, length, input_size = x.size(0), x.size(1), x.size(2)
        return torch.reshape(x, (batch_size, length // factor, input_size * factor))

    @staticmethod
    def _expand(x: torch.Tensor, factor: int) -> torch.Tensor:
        batch_size, length, input_size = x.size(0), x.size(1), x.size(2)
        return torch.reshape(x, (batch_size, length * factor, input_size // factor))


class Model(nn.Module):
    def __init__(self,
                 n_symbols: int,
                 encoder_hidden_size: int,
                 decoder_hidden_size: int,
                 reduction_factor: int):
        super(Model, self).__init__()
        self.encoder_hidden_size = encoder_hidden_size
        self.decoder_hidden_size = decoder_hidden_size
        self.reduction_factor = reduction_factor
        self.encoder = Encoder(
            n_symbols=n_symbols,
            hidden_size=encoder_hidden_size,
            n_layers=4,
            dropout=0.1
        )
        self.decoder = Decoder(
            n_mel=80,
            n_mag=513,
            hidden_size=decoder_hidden_size,
            n_layers=4,
            n_heads=1,
            n_convs=4,
            n_mag_convs=4,
            prenet_dropout=0.5,
            dropout=0.1,
            reduction_factor=reduction_factor
        )

    def infer(self,
              text: torch.Tensor,
              text_mask: torch.Tensor,
              n_features: int,
              max_length: int) -> dict:
        batch_size = text.size(0)
        spec = torch.zeros([batch_size, self.reduction_factor, n_features], dtype=torch.float32)
        length = torch.tensor(0, dtype=torch.int32)

        spec = spec.cuda()
        length = length.cuda()

        while length < max_length:
            length += self.reduction_factor
            output = self(text, text_mask, spec, length)

            mel = output['mel'][:, -self.reduction_factor:, :]
            mag = output['mag'][:, -self.reduction_factor:, :]
            spectrum = torch.cat((mel, mag), dim=-1)
            spec = torch.cat((spec, spectrum), dim=1)

        return output

    def forward(self,
                text: torch.Tensor,
                text_mask: torch.Tensor,
                spec: torch.Tensor,
                max_length: torch.Tensor) -> dict:
        memory = self.encoder(text)
        memory_mask = self._get_memory_mask(text_mask, max_length // self.reduction_factor)
        mel_spec, mag_spec, stop_token_logits, attentions = self.decoder(spec, memory, memory_mask)

        return {
            'mel': mel_spec,
            'mag': mag_spec,
            'stop_token': torch.sigmoid(stop_token_logits),
            'stop_token_logits': stop_token_logits,
            'attentions': attentions
        }

    @staticmethod
    def _get_memory_mask(text_mask: torch.Tensor, max_length: torch.Tensor) -> torch.Tensor:
        mask = text_mask.unsqueeze(-1)  # [batch, encoder, 1]
        mask = mask.repeat(1, 1, max_length)  # [batch, encoder, decoder]
        mask = mask.transpose(-1, -2)  # [batch, decoder, encoder]
        return mask
