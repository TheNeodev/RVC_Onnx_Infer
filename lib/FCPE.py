import os
import math
import torch
import librosa

import numpy as np
import soundfile as sf
import torch.nn.functional as F

from torch import nn, einsum
from functools import partial
from einops import rearrange, repeat, pack, unpack
from torch.nn.utils.parametrizations import weight_norm

os.environ["LRU_CACHE_CAPACITY"] = "3"

def exists(val):
    return val is not None

def default(value, d):
    return value if exists(value) else d

def max_neg_value(tensor):
    return -torch.finfo(tensor.dtype).max

def l2norm(tensor):
    return F.normalize(tensor, dim = -1).type(tensor.dtype)

def pad_to_multiple(tensor, multiple, dim=-1, value=0):
    seqlen = tensor.shape[dim]
    m = seqlen / multiple
    if m.is_integer(): return False, tensor
    return True, F.pad(tensor, (*((0,) * (-1 - dim) * 2), 0, (math.ceil(m) * multiple - seqlen)), value = value)

def look_around(x, backward = 1, forward = 0, pad_value = -1, dim = 2):
    t = x.shape[1]
    dims = (len(x.shape) - dim) * (0, 0)
    padded_x = F.pad(x, (*dims, backward, forward), value = pad_value)
    return torch.cat([padded_x[:, ind:(ind + t), ...] for ind in range(forward + backward + 1)], dim = dim)

def rotate_half(x):
    x1, x2 = rearrange(x, 'b ... (r d) -> b ... r d', r = 2).unbind(dim = -2)
    return torch.cat((-x2, x1), dim = -1)

def apply_rotary_pos_emb(q, k, freqs, scale = 1):
    q_len = q.shape[-2]
    q_freqs = freqs[..., -q_len:, :]
    inv_scale = scale ** -1
    if scale.ndim == 2: scale = scale[-q_len:, :]
    q = (q * q_freqs.cos() * scale) + (rotate_half(q) * q_freqs.sin() * scale)
    k = (k * freqs.cos() * inv_scale) + (rotate_half(k) * freqs.sin() * inv_scale)
    return q, k

class LocalAttention(nn.Module):
    def __init__(self, window_size, causal = False, look_backward = 1, look_forward = None, dropout = 0., shared_qk = False, rel_pos_emb_config = None, dim = None, autopad = False, exact_windowsize = False, scale = None, use_rotary_pos_emb = True, use_xpos = False, xpos_scale_base = None):
        super().__init__()
        look_forward = default(look_forward, 0 if causal else 1)
        assert not (causal and look_forward > 0)
        self.scale = scale
        self.window_size = window_size
        self.autopad = autopad
        self.exact_windowsize = exact_windowsize
        self.causal = causal
        self.look_backward = look_backward
        self.look_forward = look_forward
        self.dropout = nn.Dropout(dropout)
        self.shared_qk = shared_qk
        self.rel_pos = None
        self.use_xpos = use_xpos

        if use_rotary_pos_emb and (exists(rel_pos_emb_config) or exists(dim)): 
            if exists(rel_pos_emb_config): dim = rel_pos_emb_config[0]
            self.rel_pos = SinusoidalEmbeddings(dim, use_xpos = use_xpos, scale_base = default(xpos_scale_base, window_size // 2))

    def forward(self, q, k, v, mask = None, input_mask = None, attn_bias = None, window_size = None):
        mask = default(mask, input_mask)
        assert not (exists(window_size) and not self.use_xpos)

        _, autopad, pad_value, window_size, causal, look_backward, look_forward, shared_qk = q.shape, self.autopad, -1, default(window_size, self.window_size), self.causal, self.look_backward, self.look_forward, self.shared_qk
        (q, packed_shape), (k, _), (v, _) = map(lambda t: pack([t], '* n d'), (q, k, v))

        if autopad:
            orig_seq_len = q.shape[1]
            (_, q), (_, k), (_, v) = map(lambda t: pad_to_multiple(t, self.window_size, dim = -2), (q, k, v))

        b, n, dim_head, device, dtype = *q.shape, q.device, q.dtype
        scale = default(self.scale, dim_head ** -0.5)
        assert (n % window_size) == 0
        windows = n // window_size
        if shared_qk: k = l2norm(k)

        seq = torch.arange(n, device = device)
        b_t = rearrange(seq, '(w n) -> 1 w n', w = windows, n = window_size)
        bq, bk, bv = map(lambda t: rearrange(t, 'b (w n) d -> b w n d', w = windows), (q, k, v))
        bq = bq * scale
        look_around_kwargs = dict(backward =  look_backward, forward =  look_forward, pad_value = pad_value)
        bk = look_around(bk, **look_around_kwargs)
        bv = look_around(bv, **look_around_kwargs)

        if exists(self.rel_pos):
            pos_emb, xpos_scale = self.rel_pos(bk)
            bq, bk = apply_rotary_pos_emb(bq, bk, pos_emb, scale = xpos_scale)

        bq_t = b_t
        bq_k = look_around(b_t, **look_around_kwargs)
        bq_t = rearrange(bq_t, '... i -> ... i 1')
        bq_k = rearrange(bq_k, '... j -> ... 1 j')
        pad_mask = bq_k == pad_value
        sim = einsum('b h i e, b h j e -> b h i j', bq, bk)

        if exists(attn_bias):
            heads = attn_bias.shape[0]
            assert (b % heads) == 0
            attn_bias = repeat(attn_bias, 'h i j -> (b h) 1 i j', b = b // heads)
            sim = sim + attn_bias

        mask_value = max_neg_value(sim)

        if shared_qk:
            self_mask = bq_t == bq_k
            sim = sim.masked_fill(self_mask, -5e4)
            del self_mask

        if causal:
            causal_mask = bq_t < bq_k
            if self.exact_windowsize: causal_mask = causal_mask | (bq_t > (bq_k + (self.window_size * self.look_backward)))
            sim = sim.masked_fill(causal_mask, mask_value)
            del causal_mask

        sim = sim.masked_fill(((bq_k - (self.window_size * self.look_forward)) > bq_t) | (bq_t > (bq_k + (self.window_size * self.look_backward))) | pad_mask, mask_value) if not causal and self.exact_windowsize else sim.masked_fill(pad_mask, mask_value)

        if exists(mask):
            batch = mask.shape[0]
            assert (b % batch) == 0
            h = b // mask.shape[0]
            if autopad: _, mask = pad_to_multiple(mask, window_size, dim = -1, value = False)
            mask = repeat(rearrange(look_around(rearrange(mask, '... (w n) -> (...) w n', w = windows, n = window_size), **{**look_around_kwargs, 'pad_value': False}), '... j -> ... 1 j'), 'b ... -> (b h) ...', h = h)
            sim = sim.masked_fill(~mask, mask_value)
            del mask

        out = rearrange(einsum('b h i j, b h j e -> b h i e', self.dropout(sim.softmax(dim = -1)), bv), 'b w n d -> b (w n) d')
        if autopad: out = out[:, :orig_seq_len, :]
        out, *_ = unpack(out, packed_shape, '* n d')
        return out
    
class SinusoidalEmbeddings(nn.Module):
    def __init__(self, dim, scale_base = None, use_xpos = False, theta = 10000):
        super().__init__()
        inv_freq = 1. / (theta ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', inv_freq)
        self.use_xpos = use_xpos
        self.scale_base = scale_base
        assert not (use_xpos and not exists(scale_base))
        scale = (torch.arange(0, dim, 2) + 0.4 * dim) / (1.4 * dim)
        self.register_buffer('scale', scale, persistent = False)

    def forward(self, x):
        seq_len, device = x.shape[-2], x.device
        t = torch.arange(seq_len, device = x.device).type_as(self.inv_freq)
        freqs = torch.einsum('i , j -> i j', t, self.inv_freq)
        freqs =  torch.cat((freqs, freqs), dim = -1)

        if not self.use_xpos: return freqs, torch.ones(1, device = device)

        power = (t - (seq_len // 2)) / self.scale_base
        scale = self.scale ** rearrange(power, 'n -> n 1')
        return freqs, torch.cat((scale, scale), dim = -1)

def load_wav_to_torch(full_path, target_sr=None, return_empty_on_exception=False):
    try:
        data, sample_rate = sf.read(full_path, always_2d=True)
    except Exception as e:
        print(f"{full_path}: {e}")

        if return_empty_on_exception: return [], sample_rate or target_sr or 48000
        else: raise

    data = data[:, 0] if len(data.shape) > 1 else data
    assert len(data) > 2

    max_mag = (-np.iinfo(data.dtype).min if np.issubdtype(data.dtype, np.integer) else max(np.amax(data), -np.amin(data)))
    data = torch.FloatTensor(data.astype(np.float32)) / ((2**31) + 1 if max_mag > (2**15) else ((2**15) + 1 if max_mag > 1.01 else 1.0))

    if (torch.isinf(data) | torch.isnan(data)).any() and return_empty_on_exception: return [], sample_rate or target_sr or 48000

    if target_sr is not None and sample_rate != target_sr:
        data = torch.from_numpy(librosa.core.resample(data.numpy(), orig_sr=sample_rate, target_sr=target_sr))
        sample_rate = target_sr

    return data, sample_rate

def dynamic_range_compression(x, C=1, clip_val=1e-5):
    return np.log(np.clip(x, a_min=clip_val, a_max=None) * C)

def dynamic_range_decompression(x, C=1):
    return np.exp(x) / C

def dynamic_range_compression_torch(x, C=1, clip_val=1e-5):
    return torch.log(torch.clamp(x, min=clip_val) * C)

def dynamic_range_decompression_torch(x, C=1):
    return torch.exp(x) / C

class STFT:
    def __init__(self, sr=22050, n_mels=80, n_fft=1024, win_size=1024, hop_length=256, fmin=20, fmax=11025, clip_val=1e-5):
        self.target_sr = sr
        self.n_mels = n_mels
        self.n_fft = n_fft
        self.win_size = win_size
        self.hop_length = hop_length
        self.fmin = fmin
        self.fmax = fmax
        self.clip_val = clip_val
        self.mel_basis = {}
        self.hann_window = {}

    def get_mel(self, y, keyshift=0, speed=1, center=False, train=False):
        n_fft = self.n_fft
        win_size = self.win_size
        hop_length = self.hop_length
        fmax = self.fmax
        factor = 2 ** (keyshift / 12)
        win_size_new = int(np.round(win_size * factor))
        hop_length_new = int(np.round(hop_length * speed))
        mel_basis = self.mel_basis if not train else {}
        hann_window = self.hann_window if not train else {}
        mel_basis_key = str(fmax) + "_" + str(y.device)

        if mel_basis_key not in mel_basis:
            from librosa.filters import mel as librosa_mel_fn
            mel_basis[mel_basis_key] = torch.from_numpy(librosa_mel_fn(sr=self.target_sr, n_fft=n_fft, n_mels=self.n_mels, fmin=self.fmin, fmax=fmax)).float().to(y.device)

        keyshift_key = str(keyshift) + "_" + str(y.device)
        if keyshift_key not in hann_window: hann_window[keyshift_key] = torch.hann_window(win_size_new).to(y.device)

        pad_left = (win_size_new - hop_length_new) // 2
        pad_right = max((win_size_new - hop_length_new + 1) // 2, win_size_new - y.size(-1) - pad_left)
        spec = torch.stft(torch.nn.functional.pad(y.unsqueeze(1), (pad_left, pad_right), mode="reflect" if pad_right < y.size(-1) else "constant").squeeze(1), int(np.round(n_fft * factor)), hop_length=hop_length_new, win_length=win_size_new, window=hann_window[keyshift_key], center=center, pad_mode="reflect", normalized=False, onesided=True, return_complex=True)
        spec = torch.sqrt(spec.real.pow(2) + spec.imag.pow(2) + (1e-9))

        if keyshift != 0:
            size = n_fft // 2 + 1
            resize = spec.size(1)
            spec = (F.pad(spec, (0, 0, 0, size - resize)) if resize < size else spec[:, :size, :]) * win_size / win_size_new

        return dynamic_range_compression_torch(torch.matmul(mel_basis[mel_basis_key], spec), clip_val=self.clip_val)

    def __call__(self, audiopath):
        audio, _ = load_wav_to_torch(audiopath, target_sr=self.target_sr)
        return self.get_mel(audio.unsqueeze(0)).squeeze(0)

stft = STFT()

def softmax_kernel(data, *, projection_matrix, is_query, normalize_data=True, eps=1e-4, device=None):
    b, h, *_ = data.shape
    
    data_normalizer = (data.shape[-1] ** -0.25) if normalize_data else 1.0
    ratio = projection_matrix.shape[0] ** -0.5

    data_dash = torch.einsum("...id,...jd->...ij", (data_normalizer * data), repeat(projection_matrix, "j d -> b h j d", b=b, h=h).type_as(data))
    diag_data = ((torch.sum(data**2, dim=-1) / 2.0) * (data_normalizer**2)).unsqueeze(dim=-1)

    return (ratio * (torch.exp(data_dash - diag_data - torch.max(data_dash, dim=-1, keepdim=True).values) + eps) if is_query else ratio * (torch.exp(data_dash - diag_data + eps))).type_as(data)

def orthogonal_matrix_chunk(cols, qr_uniform_q=False, device=None):
    unstructured_block = torch.randn((cols, cols), device=device)

    q, r = torch.linalg.qr(unstructured_block.cpu(), mode="reduced")
    q, r = map(lambda t: t.to(device), (q, r))

    if qr_uniform_q:
        d = torch.diag(r, 0)
        q *= d.sign()

    return q.t()

def empty(tensor):
    return tensor.numel() == 0

def cast_tuple(val):
    return (val,) if not isinstance(val, tuple) else val

class PCmer(nn.Module):
    def __init__(self, num_layers, num_heads, dim_model, dim_keys, dim_values, residual_dropout, attention_dropout):
        super().__init__()
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dim_model = dim_model
        self.dim_values = dim_values
        self.dim_keys = dim_keys
        self.residual_dropout = residual_dropout
        self.attention_dropout = attention_dropout
        self._layers = nn.ModuleList([_EncoderLayer(self) for _ in range(num_layers)])

    def forward(self, phone, mask=None):
        for layer in self._layers:
            phone = layer(phone, mask)

        return phone

class _EncoderLayer(nn.Module):
    def __init__(self, parent: PCmer):
        super().__init__()
        self.conformer = ConformerConvModule(parent.dim_model)
        self.norm = nn.LayerNorm(parent.dim_model)
        self.dropout = nn.Dropout(parent.residual_dropout)
        self.attn = SelfAttention(dim=parent.dim_model, heads=parent.num_heads, causal=False)

    def forward(self, phone, mask=None):
        phone = phone + (self.attn(self.norm(phone), mask=mask))
        return phone + (self.conformer(phone))

def calc_same_padding(kernel_size):
    pad = kernel_size // 2
    return (pad, pad - (kernel_size + 1) % 2)

class Swish(nn.Module):
    def forward(self, x):
        return x * x.sigmoid()

class Transpose(nn.Module):
    def __init__(self, dims):
        super().__init__()
        assert len(dims) == 2, "dims == 2"

        self.dims = dims

    def forward(self, x):
        return x.transpose(*self.dims)

class GLU(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        out, gate = x.chunk(2, dim=self.dim)
        return out * gate.sigmoid()

class DepthWiseConv1d(nn.Module):
    def __init__(self, chan_in, chan_out, kernel_size, padding):
        super().__init__()
        self.padding = padding
        self.conv = nn.Conv1d(chan_in, chan_out, kernel_size, groups=chan_in)

    def forward(self, x):
        return self.conv(F.pad(x, self.padding))

class ConformerConvModule(nn.Module):
    def __init__(self, dim, causal=False, expansion_factor=2, kernel_size=31, dropout=0.0):
        super().__init__()
        inner_dim = dim * expansion_factor
        self.net = nn.Sequential(nn.LayerNorm(dim), Transpose((1, 2)), nn.Conv1d(dim, inner_dim * 2, 1), GLU(dim=1), DepthWiseConv1d(inner_dim, inner_dim, kernel_size=kernel_size, padding=(calc_same_padding(kernel_size) if not causal else (kernel_size - 1, 0))), Swish(), nn.Conv1d(inner_dim, dim, 1), Transpose((1, 2)), nn.Dropout(dropout))

    def forward(self, x):
        return self.net(x)

def linear_attention(q, k, v):
    return torch.einsum("...ed,...nd->...ne", k, q) if v is None else torch.einsum("...de,...nd,...n->...ne", torch.einsum("...nd,...ne->...de", k, v), q, 1.0 / (torch.einsum("...nd,...d->...n", q, k.sum(dim=-2).type_as(q)) + 1e-8))

def gaussian_orthogonal_random_matrix(nb_rows, nb_columns, scaling=0, qr_uniform_q=False, device=None):
    nb_full_blocks = int(nb_rows / nb_columns)
    block_list = []

    for _ in range(nb_full_blocks):
        block_list.append(orthogonal_matrix_chunk(nb_columns, qr_uniform_q=qr_uniform_q, device=device))

    remaining_rows = nb_rows - nb_full_blocks * nb_columns
    if remaining_rows > 0: block_list.append(orthogonal_matrix_chunk(nb_columns, qr_uniform_q=qr_uniform_q, device=device)[:remaining_rows])

    if scaling == 0: multiplier = torch.randn((nb_rows, nb_columns), device=device).norm(dim=1)
    elif scaling == 1: multiplier = math.sqrt((float(nb_columns))) * torch.ones((nb_rows,), device=device)
    else: raise ValueError(f"{scaling} != 0, 1")

    return torch.diag(multiplier) @ torch.cat(block_list)

class FastAttention(nn.Module):
    def __init__(self, dim_heads, nb_features=None, ortho_scaling=0, causal=False, generalized_attention=False, kernel_fn=nn.ReLU(), qr_uniform_q=False, no_projection=False):
        super().__init__()
        nb_features = default(nb_features, int(dim_heads * math.log(dim_heads)))
        self.dim_heads = dim_heads
        self.nb_features = nb_features
        self.ortho_scaling = ortho_scaling
        self.create_projection = partial(gaussian_orthogonal_random_matrix, nb_rows=self.nb_features, nb_columns=dim_heads, scaling=ortho_scaling, qr_uniform_q=qr_uniform_q)
        projection_matrix = self.create_projection()
        self.register_buffer("projection_matrix", projection_matrix)
        self.generalized_attention = generalized_attention
        self.kernel_fn = kernel_fn
        self.no_projection = no_projection
        self.causal = causal

    @torch.no_grad()
    def redraw_projection_matrix(self):
        projections = self.create_projection()
        self.projection_matrix.copy_(projections)

        del projections

    def forward(self, q, k, v):
        if self.no_projection: q, k = q.softmax(dim=-1), (torch.exp(k) if self.causal else k.softmax(dim=-2)) 
        else:
            create_kernel = partial(softmax_kernel, projection_matrix=self.projection_matrix, device=q.device)
            q, k = create_kernel(q, is_query=True), create_kernel(k, is_query=False)

        attn_fn = linear_attention if not self.causal else self.causal_linear_fn
        return attn_fn(q, k, None) if v is None else attn_fn(q, k, v)

class SelfAttention(nn.Module):
    def __init__(self, dim, causal=False, heads=8, dim_head=64, local_heads=0, local_window_size=256, nb_features=None, feature_redraw_interval=1000, generalized_attention=False, kernel_fn=nn.ReLU(), qr_uniform_q=False, dropout=0.0, no_projection=False):
        super().__init__()
        assert dim % heads == 0
        dim_head = default(dim_head, dim // heads)
        inner_dim = dim_head * heads
        self.fast_attention = FastAttention(dim_head, nb_features, causal=causal, generalized_attention=generalized_attention, kernel_fn=kernel_fn, qr_uniform_q=qr_uniform_q, no_projection=no_projection)
        self.heads = heads
        self.global_heads = heads - local_heads
        self.local_attn = (LocalAttention(window_size=local_window_size, causal=causal, autopad=True, dropout=dropout, look_forward=int(not causal), rel_pos_emb_config=(dim_head, local_heads)) if local_heads > 0 else None)
        self.to_q = nn.Linear(dim, inner_dim)
        self.to_k = nn.Linear(dim, inner_dim)
        self.to_v = nn.Linear(dim, inner_dim)
        self.to_out = nn.Linear(inner_dim, dim)
        self.dropout = nn.Dropout(dropout)

    @torch.no_grad()
    def redraw_projection_matrix(self):
        self.fast_attention.redraw_projection_matrix()

    def forward(self, x, context=None, mask=None, context_mask=None, name=None, inference=False, **kwargs):
        _, _, _, h, gh = *x.shape, self.heads, self.global_heads
        cross_attend = exists(context)

        context = default(context, x)
        context_mask = default(context_mask, mask) if not cross_attend else context_mask

        q, k, v = map(lambda t: rearrange(t, "b n (h d) -> b h n d", h=h), (self.to_q(x), self.to_k(context), self.to_v(context)))
        (q, lq), (k, lk), (v, lv) = map(lambda t: (t[:, :gh], t[:, gh:]), (q, k, v))

        attn_outs = []

        if not empty(q):
            if exists(context_mask): v.masked_fill_(~context_mask[:, None, :, None], 0.0)

            if cross_attend: pass  
            else: out = self.fast_attention(q, k, v)

            attn_outs.append(out)

        if not empty(lq):
            assert (not cross_attend), "not cross_attend"

            out = self.local_attn(lq, lk, lv, input_mask=mask)
            attn_outs.append(out)

        return self.dropout(self.to_out(rearrange(torch.cat(attn_outs, dim=1), "b h n d -> b n (h d)")))

def l2_regularization(model, l2_alpha):
    l2_loss = []

    for module in model.modules():
        if type(module) is nn.Conv2d: l2_loss.append((module.weight**2).sum() / 2.0)

    return l2_alpha * sum(l2_loss)

class _FCPE(nn.Module):
    def __init__(self, input_channel=128, out_dims=360, n_layers=12, n_chans=512, use_siren=False, use_full=False, loss_mse_scale=10, loss_l2_regularization=False, loss_l2_regularization_scale=1, loss_grad1_mse=False, loss_grad1_mse_scale=1, f0_max=1975.5, f0_min=32.70, confidence=False, threshold=0.05, use_input_conv=True):
        super().__init__()
        if use_siren: raise ValueError("Siren not support")
        if use_full: raise ValueError("Model full not support")
        self.loss_mse_scale = loss_mse_scale if (loss_mse_scale is not None) else 10
        self.loss_l2_regularization = (loss_l2_regularization if (loss_l2_regularization is not None) else False)
        self.loss_l2_regularization_scale = (loss_l2_regularization_scale if (loss_l2_regularization_scale is not None) else 1)
        self.loss_grad1_mse = loss_grad1_mse if (loss_grad1_mse is not None) else False
        self.loss_grad1_mse_scale = (loss_grad1_mse_scale if (loss_grad1_mse_scale is not None) else 1)
        self.f0_max = f0_max if (f0_max is not None) else 1975.5
        self.f0_min = f0_min if (f0_min is not None) else 32.70
        self.confidence = confidence if (confidence is not None) else False
        self.threshold = threshold if (threshold is not None) else 0.05
        self.use_input_conv = use_input_conv if (use_input_conv is not None) else True
        self.cent_table_b = torch.Tensor(np.linspace(self.f0_to_cent(torch.Tensor([f0_min]))[0], self.f0_to_cent(torch.Tensor([f0_max]))[0], out_dims))
        self.register_buffer("cent_table", self.cent_table_b)
        _leaky = nn.LeakyReLU()
        self.stack = nn.Sequential(nn.Conv1d(input_channel, n_chans, 3, 1, 1), nn.GroupNorm(4, n_chans), _leaky, nn.Conv1d(n_chans, n_chans, 3, 1, 1))
        self.decoder = PCmer(num_layers=n_layers, num_heads=8, dim_model=n_chans, dim_keys=n_chans, dim_values=n_chans, residual_dropout=0.1, attention_dropout=0.1)
        self.norm = nn.LayerNorm(n_chans)
        self.n_out = out_dims
        self.dense_out = weight_norm(nn.Linear(n_chans, self.n_out))

    def forward(self, mel, infer=True, gt_f0=None, return_hz_f0=False, cdecoder="local_argmax"):
        if cdecoder == "argmax": self.cdecoder = self.cents_decoder
        elif cdecoder == "local_argmax": self.cdecoder = self.cents_local_decoder

        x = torch.sigmoid(self.dense_out(self.norm(self.decoder((self.stack(mel.transpose(1, 2)).transpose(1, 2) if self.use_input_conv else mel)))))

        if not infer:
            loss_all = self.loss_mse_scale * F.binary_cross_entropy(x, self.gaussian_blurred_cent(self.f0_to_cent(gt_f0)))
            if self.loss_l2_regularization: loss_all = loss_all + l2_regularization(model=self, l2_alpha=self.loss_l2_regularization_scale)
            x = loss_all

        if infer:
            x = self.cent_to_f0(self.cdecoder(x))
            x = (1 + x / 700).log() if not return_hz_f0 else x

        return x

    def cents_decoder(self, y, mask=True):
        B, N, _ = y.size()
        rtn = torch.sum(self.cent_table[None, None, :].expand(B, N, -1) * y, dim=-1, keepdim=True) / torch.sum(y, dim=-1, keepdim=True)

        if mask:
            confident = torch.max(y, dim=-1, keepdim=True)[0]
            confident_mask = torch.ones_like(confident)

            confident_mask[confident <= self.threshold] = float("-INF")
            rtn = rtn * confident_mask

        return (rtn, confident) if self.confidence else rtn

    def cents_local_decoder(self, y, mask=True):
        B, N, _ = y.size()

        confident, max_index = torch.max(y, dim=-1, keepdim=True)
        local_argmax_index = torch.clamp(torch.arange(0, 9).to(max_index.device) + (max_index - 4), 0, self.n_out - 1)

        y_l = torch.gather(y, -1, local_argmax_index)
        rtn = torch.sum(torch.gather(self.cent_table[None, None, :].expand(B, N, -1), -1, local_argmax_index) * y_l, dim=-1, keepdim=True) / torch.sum(y_l, dim=-1, keepdim=True)

        if mask:
            confident_mask = torch.ones_like(confident)
            confident_mask[confident <= self.threshold] = float("-INF")

            rtn = rtn * confident_mask

        return (rtn, confident) if self.confidence else rtn

    def cent_to_f0(self, cent):
        return 10.0 * 2 ** (cent / 1200.0)

    def f0_to_cent(self, f0):
        return 1200.0 * torch.log2(f0 / 10.0)

    def gaussian_blurred_cent(self, cents):
        B, N, _ = cents.size()
        return torch.exp(-torch.square(self.cent_table[None, None, :].expand(B, N, -1) - cents) / 1250) * (cents > 0.1) & (cents < (1200.0 * np.log2(self.f0_max / 10.0))).float()

class FCPEInfer:
    def __init__(self, model_path, device=None, dtype=torch.float32, providers=None, onnx=False):
        if device is None: device = "cuda" if torch.cuda.is_available() else "cpu"
        self.wav2mel = Wav2Mel(device=device, dtype=dtype)
        self.device = device
        self.dtype = dtype
        self.onnx = onnx

        if self.onnx:
            import onnxruntime as ort

            sess_options = ort.SessionOptions()
            sess_options.log_severity_level = 3

            self.model = ort.InferenceSession(model_path, sess_options=sess_options, providers=providers)
        else:
            ckpt = torch.load(model_path, map_location=torch.device(self.device))
            self.args = DotDict(ckpt["config"])
            model = _FCPE(input_channel=self.args.model.input_channel, out_dims=self.args.model.out_dims, n_layers=self.args.model.n_layers, n_chans=self.args.model.n_chans, use_siren=self.args.model.use_siren, use_full=self.args.model.use_full, loss_mse_scale=self.args.loss.loss_mse_scale, loss_l2_regularization=self.args.loss.loss_l2_regularization, loss_l2_regularization_scale=self.args.loss.loss_l2_regularization_scale, loss_grad1_mse=self.args.loss.loss_grad1_mse, loss_grad1_mse_scale=self.args.loss.loss_grad1_mse_scale, f0_max=self.args.model.f0_max, f0_min=self.args.model.f0_min, confidence=self.args.model.confidence)

            model.to(self.device).to(self.dtype)
            model.load_state_dict(ckpt["model"])

            model.eval()
            self.model = model

    @torch.no_grad()
    def __call__(self, audio, sr, threshold=0.05):
        if not self.onnx: self.model.threshold = threshold
        mel = self.wav2mel(audio=audio[None, :], sample_rate=sr).to(self.dtype)

        return torch.as_tensor(self.model.run(["pitchf"], {"mel": mel.detach().cpu().numpy(), "threshold": np.array(threshold, dtype=np.float32)})[0], dtype=self.dtype, device=self.device).squeeze() if self.onnx else self.model(mel=mel, infer=True, return_hz_f0=True)

class Wav2Mel:
    def __init__(self, device=None, dtype=torch.float32):
        self.sample_rate = 16000
        self.hop_size = 160
        if device is None: device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.dtype = dtype
        self.stft = STFT(16000, 128, 1024, 1024, 160, 0, 8000)
        self.resample_kernel = {}

    def extract_nvstft(self, audio, keyshift=0, train=False):
        return self.stft.get_mel(audio, keyshift=keyshift, train=train).transpose(1, 2)

    def extract_mel(self, audio, sample_rate, keyshift=0, train=False):
        audio = audio.to(self.dtype).to(self.device)

        if sample_rate == self.sample_rate: audio_res = audio
        else:
            key_str = str(sample_rate)

            if key_str not in self.resample_kernel: 
                from torchaudio.transforms import Resample
                self.resample_kernel[key_str] = Resample(sample_rate, self.sample_rate, lowpass_filter_width=128)

            self.resample_kernel[key_str] = (self.resample_kernel[key_str].to(self.dtype).to(self.device))
            audio_res = self.resample_kernel[key_str](audio)

        mel = self.extract_nvstft(audio_res, keyshift=keyshift, train=train) 
        n_frames = int(audio.shape[1] // self.hop_size) + 1
        mel = (torch.cat((mel, mel[:, -1:, :]), 1) if n_frames > int(mel.shape[1]) else mel)

        return mel[:, :n_frames, :] if n_frames < int(mel.shape[1]) else mel

    def __call__(self, audio, sample_rate, keyshift=0, train=False):
        return self.extract_mel(audio, sample_rate, keyshift=keyshift, train=train)

class DotDict(dict):
    def __getattr__(*args):
        val = dict.get(*args)
        return DotDict(val) if type(val) is dict else val

    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

class FCPE:
    def __init__(self, model_path, hop_length=512, f0_min=50, f0_max=1100, dtype=torch.float32, device=None, sample_rate=44100, threshold=0.05, providers=None, onnx=False):
        self.fcpe = FCPEInfer(model_path, device=device, dtype=dtype, providers=providers, onnx=onnx)
        self.hop_length = hop_length
        self.f0_min = f0_min
        self.f0_max = f0_max
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.threshold = threshold
        self.sample_rate = sample_rate
        self.dtype = dtype
        self.name = "fcpe"

    def repeat_expand(self, content, target_len, mode = "nearest"):
        ndim = content.ndim
        content = (content[None, None] if ndim == 1 else content[None] if ndim == 2 else content)

        assert content.ndim == 3
        is_np = isinstance(content, np.ndarray)

        results = torch.nn.functional.interpolate(torch.from_numpy(content) if is_np else content, size=target_len, mode=mode)
        results = results.numpy() if is_np else results
        return results[0, 0] if ndim == 1 else results[0] if ndim == 2 else results

    def post_process(self, x, sample_rate, f0, pad_to):
        f0 = (torch.from_numpy(f0).float().to(x.device) if isinstance(f0, np.ndarray) else f0)
        f0 = self.repeat_expand(f0, pad_to) if pad_to is not None else f0

        vuv_vector = torch.zeros_like(f0)
        vuv_vector[f0 > 0.0] = 1.0
        vuv_vector[f0 <= 0.0] = 0.0

        nzindex = torch.nonzero(f0).squeeze()
        f0 = torch.index_select(f0, dim=0, index=nzindex).cpu().numpy()
        vuv_vector = F.interpolate(vuv_vector[None, None, :], size=pad_to)[0][0]

        if f0.shape[0] <= 0: return np.zeros(pad_to), vuv_vector.cpu().numpy()
        if f0.shape[0] == 1: return np.ones(pad_to) * f0[0], vuv_vector.cpu().numpy()
        
        return np.interp(np.arange(pad_to) * self.hop_length / sample_rate, self.hop_length / sample_rate * nzindex.cpu().numpy(), f0, left=f0[0], right=f0[-1]), vuv_vector.cpu().numpy()

    def compute_f0(self, wav, p_len=None):
        x = torch.FloatTensor(wav).to(self.dtype).to(self.device)
        p_len = x.shape[0] // self.hop_length if p_len is None else p_len

        f0 = self.fcpe(x, sr=self.sample_rate, threshold=self.threshold)
        f0 = f0[:] if f0.dim() == 1 else f0[0, :, 0]

        if torch.all(f0 == 0): return f0.cpu().numpy() if p_len is None else np.zeros(p_len), (f0.cpu().numpy() if p_len is None else np.zeros(p_len))
        return self.post_process(x, self.sample_rate, f0, p_len)[0]