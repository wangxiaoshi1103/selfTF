import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.transformer.multiHeadAttention import MultiHeadAttention, MultiHeadAttentionChunk, MultiHeadAttentionWindow
from models.transformer.positionwiseFeedForward import PositionwiseFeedForward
from models.semantic_fusion_ts.gated_residual_network import GatedResidualNetwork


class Encoder(nn.Module):
    """Encoder block from Attention is All You Need.

    Apply Multi Head Attention block followed by a Point-wise Feed Forward block.
    Residual sum and normalization are applied at each step.

    Parameters
    ----------
    d_model:
        Dimension of the input vector.
    q:
        Dimension of all query matrix.
    v:
        Dimension of all value matrix.
    h:
        Number of heads.
    attention_size:
        Number of backward elements to apply attention.
        Deactivated if ``None``. Default is ``None``.
    dropout:
        Dropout probability after each MHA or PFF block.
        Default is ``0.3``.
    chunk_mode:
        Swict between different MultiHeadAttention blocks.
        One of ``'chunk'``, ``'window'`` or ``None``. Default is ``'chunk'``.
    """

    def __init__(self,
                 d_model: int,
                 q: int,
                 v: int,
                 h: int,
                 attention_size: int = None,
                 dropout: float = 0.3,
                 chunk_mode: str = 'chunk'):
        """Initialize the Encoder block"""
        super().__init__()

        chunk_mode_modules = {
            'chunk': MultiHeadAttentionChunk,
            'window': MultiHeadAttentionWindow,
        }

        if chunk_mode in chunk_mode_modules.keys():
            MHA = chunk_mode_modules[chunk_mode]
        else:
            MHA = MultiHeadAttention

        self._selfAttention = MHA(d_model, q, v, h, attention_size=attention_size)
        self._feedForward = PositionwiseFeedForward(d_model)

        self._layerNorm1 = nn.LayerNorm(d_model)
        self._layerNorm2 = nn.LayerNorm(d_model)

        self._dopout = nn.Dropout(p=dropout)

        self.grn = GatedResidualNetwork(
            input_size=d_model,
            hidden_layer_size=d_model,
            output_size=None,
            dropout_rate=0.1,
            use_time_distributed=True,
            return_gate=True,
            batch_first=True)

    def forward(self, x: torch.Tensor, context=None) -> torch.Tensor:
        """Propagate the input through the Encoder block.

        Apply the Multi Head Attention block, add residual and normalize.
        Apply the Point-wise Feed Forward block, add residual and normalize.

        Parameters
        ----------
        x:
            Input tensor with shape (batch_size, K, d_model).

        Returns
        -------
            Output tensor with shape (batch_size, K, d_model).
        """
        # Self attention
        residual = x
        x = self._selfAttention(query=x, key=x, value=x)
        x = self._dopout(x)
        x = self._layerNorm1(x + residual)

        # Inject static vars
        if context is not None:
            x, _ = self.grn(x, context)

        # Feed forward
        residual = x
        x = self._feedForward(x)
        x = self._dopout(x)
        x = self._layerNorm2(x + residual)

        return x

    @property
    def attention_map(self) -> torch.Tensor:
        """Attention map after a forward propagation,
        variable `score` in the original paper.
        """
        return self._selfAttention.attention_map
