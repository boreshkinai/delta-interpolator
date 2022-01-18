import math

import torch
import torch as t
import torch.nn.functional as f
from typing import Tuple

from torch import nn as nn
from torch.nn import functional as F


class FCBlock(t.nn.Module):
    """Fully connected residual block"""

    def __init__(self, num_layers: int, layer_width: int, dropout: float, size_in: int, size_out: int):
        super(FCBlock, self).__init__()
        self.num_layers = num_layers
        self.layer_width = layer_width

        if num_layers > 1:
            self.fc_layers = [t.nn.Linear(size_in, layer_width)]
            self.relu_layers = [t.nn.LeakyReLU(inplace=True)]
            if dropout > 0.0:
                self.fc_layers.append(t.nn.Dropout(p=dropout))
                self.relu_layers.append(t.nn.Identity())
            self.fc_layers += [t.nn.Linear(layer_width, layer_width) for _ in range(num_layers - 1)]
            self.relu_layers += [t.nn.LeakyReLU(inplace=True) for _ in range(num_layers - 1)]

            self.forward_projection = t.nn.Linear(layer_width, size_out)
            self.backward_projection = t.nn.Linear(size_in, layer_width)
            self.fc_layers = t.nn.ModuleList(self.fc_layers)
            self.relu_layers = t.nn.ModuleList(self.relu_layers)
        else:
            self.fc_layers = []
            self.relu_layers = []
            self.forward_projection = t.nn.Linear(size_in, size_out)
            self.backward_projection = t.nn.Identity()

    def forward(self, x: t.Tensor) -> Tuple[t.Tensor, t.Tensor]:
        h = x
        for layer, relu in zip(self.fc_layers, self.relu_layers):
            h = relu(layer(h))
        f = self.forward_projection(h)
        b = t.relu(h + self.backward_projection(x))
        return b, f


class LayerNorm(t.nn.Module):
    "Construct a layernorm module (See citation for details)."

    def __init__(self, num_features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = t.nn.Parameter(t.ones(num_features))
        self.b_2 = t.nn.Parameter(t.zeros(num_features))
        self.eps = eps

    def forward(self, x: t.Tensor) -> t.Tensor:
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2


class Embedding(t.nn.Module):
    """Implementation of embedding using one hot encoded input and fully connected layer"""

    def __init__(self, num_embeddings: int, embedding_dim: int):
        super(Embedding, self).__init__()
        self.projection = t.nn.Linear(num_embeddings, embedding_dim)
        self.num_embeddings = num_embeddings

    def forward(self, e: t.Tensor) -> t.Tensor:
        e_ohe = f.one_hot(e, num_classes=self.num_embeddings).float()
        return self.projection(e_ohe)


class MultiHeadAttention(nn.Module):

    def __init__(self,
                 in_features,
                 head_num,
                 bias=True,
                 activation=F.relu):
        """Multi-head attention.

        :param in_features: Size of each input sample.
        :param head_num: Number of heads.
        :param bias: Whether to use the bias term.
        :param activation: The activation after each linear transformation.
        """
        super(MultiHeadAttention, self).__init__()
        if in_features % head_num != 0:
            raise ValueError('`in_features`({}) should be divisible by `head_num`({})'.format(in_features, head_num))
        self.in_features = in_features
        self.head_num = head_num
        self.activation = activation
        self.bias = bias
        self.linear_q = nn.Linear(in_features, in_features, bias)
        self.linear_k = nn.Linear(in_features, in_features, bias)
        self.linear_v = nn.Linear(in_features, in_features, bias)
        self.linear_o = nn.Linear(in_features, in_features, bias)

    def forward(self, q, k, v, mask=None):
        q, k, v = self.linear_q(q), self.linear_k(k), self.linear_v(v)
        if self.activation is not None:
            q = self.activation(q)
            k = self.activation(k)
            v = self.activation(v)

        q = self._reshape_to_batches(q)
        k = self._reshape_to_batches(k)
        v = self._reshape_to_batches(v)
        if mask is not None:
            mask = mask.repeat(self.head_num, 1, 1)
        y = ScaledDotProductAttention()(q, k, v, mask)
        y = self._reshape_from_batches(y)

        y = self.linear_o(y)
        if self.activation is not None:
            y = self.activation(y)
        return y

    @staticmethod
    def gen_history_mask(x):
        """Generate the mask that only uses history data.

        :param x: Input tensor.
        :return: The mask.
        """
        batch_size, seq_len, _ = x.size()
        return torch.tril(torch.ones(seq_len, seq_len)).view(1, seq_len, seq_len).repeat(batch_size, 1, 1)

    def _reshape_to_batches(self, x):
        batch_size, seq_len, in_feature = x.size()
        sub_dim = in_feature // self.head_num
        return x.reshape(batch_size, seq_len, self.head_num, sub_dim)\
                .permute(0, 2, 1, 3)\
                .reshape(batch_size * self.head_num, seq_len, sub_dim)

    def _reshape_from_batches(self, x):
        batch_size, seq_len, in_feature = x.size()
        batch_size //= self.head_num
        out_dim = in_feature * self.head_num
        return x.reshape(batch_size, self.head_num, seq_len, in_feature)\
                .permute(0, 2, 1, 3)\
                .reshape(batch_size, seq_len, out_dim)

    def extra_repr(self):
        return 'in_features={}, head_num={}, bias={}, activation={}'.format(
            self.in_features, self.head_num, self.bias, self.activation,
        )


class ScaledDotProductAttention(nn.Module):
    def forward(self, query, key, value, mask=None):
        dk = query.size()[-1]
        scores = query.matmul(key.transpose(-2, -1)) / math.sqrt(dk)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        attention = F.softmax(scores, dim=-1)
        return attention.matmul(value)