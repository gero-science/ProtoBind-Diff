import math
from math import pi
import typing
from typing import Tuple, Optional, Literal

from einops import rearrange, repeat

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import autocast
from torch.nn import Module, ModuleList
from torch import nn, einsum, broadcast_tensors, Tensor



#################################################################################
#                                Rotary Encoding                                #
#################################################################################

# helper functions

def exists(val):
    return val is not None


def default(val, d):
    return val if exists(val) else d

def slice_at_dim(t, dim_slice: slice, *, dim):
    dim += (t.ndim if dim < 0 else 0)
    colons = [slice(None)] * t.ndim
    colons[dim] = dim_slice
    return t[tuple(colons)]

# rotary embedding helper functions

def rotate_half(x):
    """Splits the last dimension of a tensor, swaps halves, and negates the first half."""
    x = rearrange(x, '... (d r) -> ... d r', r=2)
    x1, x2 = x.unbind(dim=-1)
    x = torch.stack((-x2, x1), dim=-1)
    return rearrange(x, '... d r -> ... (d r)')


@autocast('cuda', enabled=False)
def apply_rotary_emb(
        freqs,
        t,
        start_index=0,
        scale=1.,
        seq_dim=-2,
        freqs_seq_dim=None
):
    """Applies rotary positional embeddings to a given tensor.

        Args:
            freqs (torch.Tensor): The rotary frequencies.
            t (torch.Tensor): The tensor to apply embeddings to (e.g., queries or keys).
            start_index (int): The feature dimension index to start applying rotations from.
            scale (float): A scaling factor, used for xPos.
            seq_dim (int): The sequence dimension of the input tensor `t`.
            freqs_seq_dim (Optional[int]): The sequence dimension of the freqs tensor.
    """
    dtype = t.dtype

    if not exists(freqs_seq_dim):
        if freqs.ndim == 2 or t.ndim == 3:
            freqs_seq_dim = 0

    if t.ndim == 3 or exists(freqs_seq_dim):
        seq_len = t.shape[seq_dim]
        freqs = slice_at_dim(freqs, slice(-seq_len, None), dim=freqs_seq_dim)

    rot_dim = freqs.shape[-1]
    end_index = start_index + rot_dim

    assert rot_dim <= t.shape[
        -1], f'feature dimension {t.shape[-1]} is not of sufficient size to rotate in all the positions {rot_dim}'

    # Split t into three parts: left, middle (to be transformed), and right
    t_left = t[..., :start_index]
    t_middle = t[..., start_index:end_index]
    t_right = t[..., end_index:]

    # Apply rotary embeddings without modifying t in place
    t_transformed = (t_middle * freqs.cos() * scale) + (rotate_half(t_middle) * freqs.sin() * scale)

    out = torch.cat((t_left, t_transformed, t_right), dim=-1)

    return out.type(dtype)


# learned rotation helpers

def apply_learned_rotations(rotations, t, start_index=0, freq_ranges=None):
    if exists(freq_ranges):
        rotations = einsum('..., f -> ... f', rotations, freq_ranges)
        rotations = rearrange(rotations, '... r f -> ... (r f)')

    rotations = repeat(rotations, '... n -> ... (n r)', r=2)
    return apply_rotary_emb(rotations, t, start_index=start_index)


# classes

class RotaryEmbedding(Module):
    """
    original paper: https://arxiv.org/abs/2104.09864
    rescale rotary embeddings to longer sequence length without fine-tuning
    code source: https://github.com/lucidrains/rotary-embedding-torch
    """

    def __init__(
            self,
            dim,
            custom_freqs: Tensor | None = None,
            freqs_for: Literal['lang', 'pixel', 'constant'] = 'lang',
            theta=10000,
            max_freq=10,
            num_freqs=1,
            learned_freq=False,
            use_xpos=False,
            xpos_scale_base=512,
            interpolate_factor=1.,
            theta_rescale_factor=1.,
            seq_before_head_dim=False,
            cache_if_possible=True,
            cache_max_seq_len=8192
    ):
        super().__init__()
        """Initializes the RotaryEmbedding module.

                Args:
                    dim (int): The feature dimension to apply rotary embeddings to.
                    custom_freqs ([Tensor]): An optional tensor of custom frequencies.
                    freqs_for : The method for generating
                        frequencies. 'lang' is standard for transformers.
                    theta (int): A core hyperparameter for frequency calculation.
                    learned_freq (bool): If True, the frequencies are trainable parameters.
                    use_xpos (bool): If True, enables the xPos (extrapolatable) variant.
                    interpolate_factor (float): A factor for positional interpolation, which
                        can help with length generalization.
                    cache_if_possible (bool): If True, caches calculated frequencies for efficiency.
        """
        theta *= theta_rescale_factor ** (dim / (dim - 2))

        self.freqs_for = freqs_for

        if exists(custom_freqs):
            freqs = custom_freqs
        elif freqs_for == 'lang':
            freqs = 1. / (theta ** (torch.arange(0, dim, 2)[:(dim // 2)].float() / dim))
        elif freqs_for == 'pixel':
            freqs = torch.linspace(1., max_freq / 2, dim // 2) * pi
        elif freqs_for == 'constant':
            freqs = torch.ones(num_freqs).float()

        self.cache_if_possible = cache_if_possible
        self.cache_max_seq_len = cache_max_seq_len

        self.register_buffer('cached_freqs', torch.zeros(cache_max_seq_len, dim), persistent=False)
        self.cached_freqs_seq_len = 0

        self.freqs = nn.Parameter(freqs, requires_grad=learned_freq)

        self.learned_freq = learned_freq

        # dummy for device

        self.register_buffer('dummy', torch.tensor(0), persistent=False)

        # default sequence dimension

        self.seq_before_head_dim = seq_before_head_dim
        self.default_seq_dim = -3 if seq_before_head_dim else -2

        # interpolation factors

        assert interpolate_factor >= 1.
        self.interpolate_factor = interpolate_factor

        # xpos

        self.use_xpos = use_xpos

        if not use_xpos:
            return

        scale = (torch.arange(0, dim, 2) + 0.4 * dim) / (1.4 * dim)
        self.scale_base = xpos_scale_base

        self.register_buffer('scale', scale, persistent=False)
        self.register_buffer('cached_scales', torch.zeros(cache_max_seq_len, dim), persistent=False)
        self.cached_scales_seq_len = 0

        # add apply_rotary_emb as static method

        self.apply_rotary_emb = staticmethod(apply_rotary_emb)

    @property
    def device(self):
        return self.dummy.device

    def get_seq_pos(self, seq_len, device, dtype, offset=0):
        return (torch.arange(seq_len, device=device, dtype=dtype) + offset) / self.interpolate_factor

    def rotate_queries_or_keys(self, t, seq_dim=None, offset=0, scale=None):
        """Applies rotary embeddings to a single tensor (queries or keys).

                Args:
                    t (torch.Tensor): The input tensor (queries or keys).
                    seq_dim : The sequence dimension of the tensor.
                    offset (int): An offset for the position sequence, used for caching.
                    scale (Optional[float]): A scaling factor, required if using xPos.

                Returns:
                    torch.Tensor: The tensor with rotary embeddings applied.
         """
        seq_dim = default(seq_dim, self.default_seq_dim)

        assert not self.use_xpos or exists(
            scale), 'you must use `.rotate_queries_and_keys` method instead and pass in both queries and keys, for length extrapolatable rotary embeddings'

        device, dtype, seq_len = t.device, t.dtype, t.shape[seq_dim]

        seq = self.get_seq_pos(seq_len, device=device, dtype=dtype, offset=offset)

        freqs = self.forward(seq, seq_len=seq_len, offset=offset)

        if seq_dim == -3:
            freqs = rearrange(freqs, 'n d -> n 1 d')

        return apply_rotary_emb(freqs, t, scale=default(scale, 1.), seq_dim=seq_dim)

    def rotate_queries_with_cached_keys(self, q, k, seq_dim=None, offset=0):
        dtype, device, seq_dim = q.dtype, q.device, default(seq_dim, self.default_seq_dim)

        q_len, k_len = q.shape[seq_dim], k.shape[seq_dim]
        assert q_len <= k_len

        q_scale = k_scale = 1.

        if self.use_xpos:
            seq = self.get_seq_pos(k_len, dtype=dtype, device=device)

            q_scale = self.get_scale(seq[-q_len:]).type(dtype)
            k_scale = self.get_scale(seq).type(dtype)

        rotated_q = self.rotate_queries_or_keys(q, seq_dim=seq_dim, scale=q_scale, offset=k_len - q_len + offset)
        rotated_k = self.rotate_queries_or_keys(k, seq_dim=seq_dim, scale=k_scale ** -1)

        rotated_q = rotated_q.type(q.dtype)
        rotated_k = rotated_k.type(k.dtype)

        return rotated_q, rotated_k

    def rotate_queries_and_keys(self, q, k, seq_dim=None):
        seq_dim = default(seq_dim, self.default_seq_dim)

        assert self.use_xpos
        device, dtype, seq_len = q.device, q.dtype, q.shape[seq_dim]

        seq = self.get_seq_pos(seq_len, dtype=dtype, device=device)

        freqs = self.forward(seq, seq_len=seq_len)
        scale = self.get_scale(seq, seq_len=seq_len).to(dtype)

        if seq_dim == -3:
            freqs = rearrange(freqs, 'n d -> n 1 d')
            scale = rearrange(scale, 'n d -> n 1 d')

        rotated_q = apply_rotary_emb(freqs, q, scale=scale, seq_dim=seq_dim)
        rotated_k = apply_rotary_emb(freqs, k, scale=scale ** -1, seq_dim=seq_dim)

        rotated_q = rotated_q.type(q.dtype)
        rotated_k = rotated_k.type(k.dtype)

        return rotated_q, rotated_k

    def get_scale(
            self,
            t: Tensor,
            seq_len = None,
            offset=0
    ):
        assert self.use_xpos

        should_cache = (
                self.cache_if_possible and
                exists(seq_len) and
                (offset + seq_len) <= self.cache_max_seq_len
        )

        if (
                should_cache and \
                exists(self.cached_scales) and \
                (seq_len + offset) <= self.cached_scales_seq_len
        ):
            return self.cached_scales[offset:(offset + seq_len)]

        scale = 1.
        if self.use_xpos:
            power = (t - len(t) // 2) / self.scale_base
            scale = self.scale ** rearrange(power, 'n -> n 1')
            scale = repeat(scale, 'n d -> n (d r)', r=2)

        if should_cache and offset == 0:
            self.cached_scales[:seq_len] = scale.detach()
            self.cached_scales_seq_len = seq_len

        return scale

    def get_axial_freqs(self, *dims):
        Colon = slice(None)
        all_freqs = []

        for ind, dim in enumerate(dims):
            if self.freqs_for == 'pixel':
                pos = torch.linspace(-1, 1, steps=dim, device=self.device)
            else:
                pos = torch.arange(dim, device=self.device)

            freqs = self.forward(pos, seq_len=dim)

            all_axis = [None] * len(dims)
            all_axis[ind] = Colon

            new_axis_slice = (Ellipsis, *all_axis, Colon)
            all_freqs.append(freqs[new_axis_slice])

        all_freqs = broadcast_tensors(*all_freqs)
        return torch.cat(all_freqs, dim=-1)

    @autocast('cuda', enabled=False)
    def forward(
            self,
            t: Tensor,
            seq_len = None,
            offset=0
    ):
        """Calculates the rotary frequencies for a given sequence of positions.

        Args:
            t (torch.Tensor): A tensor of position indices.
            seq_len (int): The total sequence length, used for caching.
            offset (int): The starting position offset.

        Returns:
            torch.Tensor: A tensor of calculated rotation frequencies.
        """
        should_cache = (
                self.cache_if_possible and
                not self.learned_freq and
                exists(seq_len) and
                self.freqs_for != 'pixel' and
                (offset + seq_len) <= self.cache_max_seq_len
        )

        if (
                should_cache and \
                exists(self.cached_freqs) and \
                (offset + seq_len) <= self.cached_freqs_seq_len
        ):
            return self.cached_freqs[offset:(offset + seq_len)].detach()

        freqs = self.freqs

        freqs = einsum('..., f -> ... f', t.type(freqs.dtype), freqs)
        freqs = repeat(freqs, '... n -> ... (n r)', r=2)

        if should_cache and offset == 0:
            self.cached_freqs[:seq_len] = freqs.detach()
            self.cached_freqs_seq_len = seq_len

        return freqs


#################################################################################
#                             Multi Head Attention                              #
#################################################################################

class LayerNorm(nn.Module):
    """Implements a Layer Normalization module."""
    def __init__(self, d_model, eps=1e-12):
        """Initializes the LayerNorm module.

        Args:
            d_model (int): The dimension of the model's features.
            eps (float): A small value added to the variance for numerical stability.
        """
        super(LayerNorm, self).__init__()
        self.gamma = nn.Parameter(torch.ones(d_model))
        self.beta = nn.Parameter(torch.zeros(d_model))
        self.eps = eps

    def forward(self, x):
        """Applies Layer Normalization to the input tensor along the last dimension.
        Args:
            x (torch.Tensor): The input tensor to normalize.
        Returns:
            torch.Tensor: The normalized tensor.
        """
        mean = x.mean(-1, keepdim=True)
        var = x.var(-1, unbiased=False, keepdim=True)
        # '-1' means last dimension.

        out = (x - mean) / torch.sqrt(var + self.eps)
        out = self.gamma * out + self.beta
        return out


class PositionwiseFeedForward(nn.Module):
    """Implements the Position-wise Feed-Forward network of a Transformer block."""

    def __init__(self, d_model, hidden, drop_prob=0.1):
        """Initializes the PositionwiseFeedForward module.

        Args:
            d_model (int): The input and output dimension of the layer.
            hidden (int): The dimension of the inner hidden layer.
            drop_prob (float): The probability for the dropout layer.
        """
        super(PositionwiseFeedForward, self).__init__()
        self.linear1 = nn.Linear(d_model, hidden)
        self.linear2 = nn.Linear(hidden, d_model)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=drop_prob)

    def forward(self, x):
        """Passes the input through the feed-forward network.
        The process is: Linear -> ReLU -> Dropout -> Linear.
        Args:
            x (torch.Tensor): The input tensor.
        Returns:
            torch.Tensor: The output tensor.
        """
        x = self.linear1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.linear2(x)
        return x


class ScaleDotProductAttention(nn.Module):

    def __init__(self):
        super(ScaleDotProductAttention, self).__init__()
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, q, k, v, mask=None, e=1e-12):
        """
        Performs the Scaled Dot-Product Attention calculation.

        Args:
            q (torch.Tensor): The query tensor.
            k (torch.Tensor): The key tensor.
            v (torch.Tensor): The value tensor.
            mask (torch.Tensor, optional): A mask to prevent attention to
                certain positions. Defaults to None.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: A tuple containing the attention
            output and the attention scores.
        """
        batch_size, head, length, d_tensor = k.size()
        k_t = k.transpose(2, 3)  # transpose
        score = (q @ k_t) / math.sqrt(d_tensor)  # scaled dot product
        if mask is not None:
            score = score.masked_fill(mask == 0, -10000)
        score = self.softmax(score)
        v = score @ v
        return v, score


class MultiHeadAttention(nn.Module):
    """Implements a Multi-Head Attention layer with optional Rotary Position Embeddings."""

    def __init__(self, d_model, n_head):
        """Initializes the MultiHeadAttention layer.

          Args:
              d_model (int): The total dimension of the model.
              n_head (int): The number of attention heads. d_model must be divisible by n_head.
        """
        super(MultiHeadAttention, self).__init__()
        self.n_head = n_head
        self.attention = ScaleDotProductAttention()
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_concat = nn.Linear(d_model, d_model)

        self.rotary_emb = RotaryEmbedding(dim=d_model // n_head)

    def forward(self, q, k, v, mask=None, apply_rotary=False):
        """Performs the forward pass for multi-head attention.

                Args:
                    q (torch.Tensor): The query tensor.
                    k (torch.Tensor): The key tensor.
                    v (torch.Tensor): The value tensor.
                    mask (torch.Tensor, optional): An attention mask. Defaults to None.
                    apply_rotary (bool): If True, applies Rotary Position Embeddings to Q and K
                        before the attention calculation. Defaults to False.

                Returns:
                    Tuple[torch.Tensor, torch.Tensor]: A tuple containing the final output tensor
                    and the attention scores.
        """
        q, k, v = self.w_q(q), self.w_k(k), self.w_v(v)
        q, k, v = self.split(q), self.split(k), self.split(v)

        if apply_rotary:
            # add Rotary Positional Embeddings (RoPE)
            # https://arxiv.org/abs/2104.09864
            q = self.rotary_emb.rotate_queries_or_keys(q)
            k = self.rotary_emb.rotate_queries_or_keys(k)

        out, attention = self.attention(q, k, v, mask=mask)
        out = self.concat(out)
        out = self.w_concat(out)
        return out, attention

    def split(self, tensor):
        """Splits the last dimension of a tensor into multiple heads."""
        batch_size, length, d_model = tensor.size()
        d_tensor = d_model // self.n_head
        tensor = tensor.view(batch_size, length, self.n_head, d_tensor).transpose(1, 2)
        return tensor

    def concat(self, tensor):
        """Concatenates multiple heads back into a single tensor."""
        batch_size, head, length, d_tensor = tensor.size()
        d_model = head * d_tensor
        tensor = tensor.transpose(1, 2).contiguous().view(batch_size, length, d_model)
        return tensor


#################################################################################
#                               Embedding Layers                                #
#################################################################################

class EmbeddingLayer(nn.Module):
    """A simple lookup-based embedding layer with Kaiming uniform initialization."""
    def __init__(self, dim, vocab_dim):
        super().__init__()
        self.embedding = nn.Parameter(torch.empty((vocab_dim, dim)))
        torch.nn.init.kaiming_uniform_(self.embedding, a=math.sqrt(5))

    def forward(self, x):
        """Looks up the embeddings for the given indices.
        Args:
            x (torch.Tensor): A tensor of integer indices.
        Returns:
            torch.Tensor: The corresponding embedding vectors.
        """
        return self.embedding[x]


class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """

    def __init__(self, hidden_size, frequency_embedding_size=256):
        """Initializes the TimestepEmbedder.

         Args:
             hidden_size (int): The final dimension of the timestep embedding.
             frequency_embedding_size (int): The number of frequencies to use for
                 the sinusoidal embedding.
         """
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True))
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
                          These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
        half = dim // 2
        freqs = torch.exp(
            - math.log(max_period)
            * torch.arange(start=0, end=half, dtype=torch.float32)
            / half).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat(
                [embedding,
                 torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb


#################################################################################
#                                  Decoder                                      #
#################################################################################

class DecoderLayer(nn.Module):
    """
    code source: https://github.com/hyunwoongko/transformer
    """

    def __init__(self, d_model, ffn_hidden, n_head, drop_prob):
        """Initializes the DecoderLayer.

        Args:
            d_model (int): The dimension of the model.
            ffn_hidden (int): The dimension of the hidden layer in the feed-forward network.
            n_head (int): The number of attention heads.
            drop_prob (float): The dropout probability.
        """
        super(DecoderLayer, self).__init__()

        self.self_attention = MultiHeadAttention(d_model=d_model, n_head=n_head)
        self.norm1 = LayerNorm(d_model=d_model)
        self.dropout1 = nn.Dropout(p=drop_prob)

        self.enc_dec_attention = MultiHeadAttention(d_model=d_model, n_head=n_head)
        self.norm2 = LayerNorm(d_model=d_model)
        self.dropout2 = nn.Dropout(p=drop_prob)

        self.ffn = PositionwiseFeedForward(d_model=d_model, hidden=ffn_hidden, drop_prob=drop_prob)
        self.norm3 = LayerNorm(d_model=d_model)
        self.dropout3 = nn.Dropout(p=drop_prob)

    def forward(self, dec, enc, trg_mask, src_mask, return_attention=False):
        """Performs one forward pass of the decoder layer.

        Args:
            dec (torch.Tensor): The input tensor from the previous decoder layer.
            enc (torch.Tensor): The output tensor from the encoder (for conditioning).
            trg_mask (torch.Tensor): The mask for the decoder's self-attention.
            src_mask (torch.Tensor): The mask for the cross-attention.
            return_attention (bool): If True, returns the cross-attention weights.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: A tuple containing the output tensor
            and the attention weights (or None).
        """
        attention = None

        _x = dec
        x, _ = self.self_attention(q=dec, k=dec, v=dec, mask=trg_mask, apply_rotary=True)
        x = self.dropout1(x)
        x = self.norm1(x + _x)

        if enc is not None:
            _x = x
            if return_attention:
                x, attention = self.enc_dec_attention(q=x, k=enc, v=enc, mask=src_mask)
            else:
                x, _ = self.enc_dec_attention(q=x, k=enc, v=enc, mask=src_mask)
            x = self.dropout2(x)
            x = self.norm2(x + _x)

        _x = x
        x = self.ffn(x)
        x = self.dropout3(x)
        x = self.norm3(x + _x)
        return x, attention


class Decoder_RoPE(nn.Module):
    """A decoder that uses Rotary Position Embeddings (RoPE).

    This model is designed for a diffusion task, taking a ligand sequence, a
    conditioning protein sequence, and a diffusion timestep (sigma) as input
    to predict the output logits for the ligand.
    """
    def __init__(self,
                 vocab_size,
                 seq_emb_dim,
                 hidden_size: int=640,
                 nhead: int=8,
                 n_layers: int=4,
                 expand_feedforward: int=3,
                 dropout: float=0.1):

        """Args:
            vocab_size (int): The size of the output vocabulary (e.g., ligand tokens).
            seq_emb_dim (int): The dimension of the input sequence embeddings.
            hidden_size (int): The main hidden dimension of the Transformer model.
            nhead (int): The number of attention heads in each DecoderLayer.
            n_layers (int): The number of DecoderLayers to stack.
            expand_feedforward (int): The expansion factor for the feed-forward
                network's hidden layer.
            dropout (float): The dropout probability.
        """
        super().__init__()

        self.hidden_size = hidden_size
        self.vocab_embed = EmbeddingLayer(self.hidden_size, vocab_size)
        self.linear = nn.Linear(self.hidden_size, vocab_size)
        self.apply_seq_linear = False

        if seq_emb_dim != self.hidden_size:
            self.apply_seq_linear = True
            self.linear_seq = nn.Linear(seq_emb_dim, self.hidden_size)

        self.sigma_map = TimestepEmbedder(self.hidden_size)

        self.layers = nn.ModuleList([DecoderLayer(d_model=self.hidden_size,
                                                  ffn_hidden=self.hidden_size * expand_feedforward,
                                                  n_head=nhead,
                                                  drop_prob=dropout)
                                     for _ in range(n_layers)])

    def forward(self,
                ligand: torch.Tensor,
                sigma: torch.Tensor,
                sequence: torch.Tensor,
                sequence_lengths: torch.Tensor,
                lig_padding_mask: Optional[torch.Tensor]=None,
                return_attention: bool=False) -> Tuple[torch.Tensor, torch.Tensor]:
        """Performs the forward pass of the decoder.

        It processes the ligand sequence conditioned on the protein sequence and the
        diffusion timestep (sigma). The sigma embedding is prepended to the protein
        sequence to form a single conditioning context.

        Args:
            ligand (torch.Tensor): A batch of ligand token ID tensors.
            sigma (torch.Tensor): A batch of scalar diffusion timesteps.
            sequence (torch.Tensor): A batch of conditioning protein sequence embeddings.
            sequence_lengths (torch.Tensor): The original lengths of the protein sequences.
            lig_padding_mask (Optional[torch.Tensor]): A padding mask for the ligand.
            return_attention (bool): If True, returns the cross-attention weights
                from the last decoder layer.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: A tuple of (output_logits, attention_weights).
        """
        ligand = self.vocab_embed(ligand)
        sigma = F.silu(self.sigma_map(sigma)).unsqueeze(1)
        if self.apply_seq_linear:
            sequence = self.linear_seq(sequence)
        condition = torch.cat([sigma, sequence], dim=1)
        sequence_lengths += 1

        range_tensor = torch.arange(condition.shape[1], device=sequence.device).unsqueeze(0)
        condition_mask = range_tensor < sequence_lengths.unsqueeze(1)
        condition_mask = condition_mask.unsqueeze(1).unsqueeze(2)
        if lig_padding_mask is not None:
            lig_padding_mask = lig_padding_mask.unsqueeze(1).unsqueeze(2)

        for layer in self.layers:
            ligand, attention = layer(ligand, condition,
                                      trg_mask=lig_padding_mask, src_mask=condition_mask,
                                      return_attention=return_attention)

        output = self.linear(ligand)
        return output, attention
