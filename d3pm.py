"""Masked D3PM inference extracted from Simple_Character_D3PM.ipynb.

Model and sampler definitions retain the notebook's state-dict layout and schedule.
CheckpointStore retains read/selection/recovery only; inference never writes a run.
"""

import hashlib
import json
import math
from pathlib import Path
import shutil
import tempfile

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

FORMAT_VERSION = 1


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()

def read_payload(manifest_path):
    manifest_path = Path(manifest_path)
    info = json.loads(manifest_path.read_text(encoding='utf-8'))
    if info['format_version'] != FORMAT_VERSION:
        raise ValueError('Unsupported artifact version.')
    filename = info['file']
    if Path(filename).name != filename or not filename.endswith('.pt'):
        raise ValueError('Invalid payload filename.')
    source = manifest_path.parent / filename
    if source.stat().st_size != info['bytes']:
        raise IOError('Size mismatch: ' + str(source))
    with tempfile.TemporaryDirectory(prefix='chard3pm-load-') as temporary:
        local = Path(temporary) / 'payload.pt'
        shutil.copyfile(source, local)
        if sha256_file(local) != info['sha256']:
            raise IOError('Checksum mismatch: ' + str(source))
        payload = torch.load(local, map_location='cpu', weights_only=True)
    return payload, info

class CheckpointStore:

    def __init__(self, run_dir):
        self.folder = Path(run_dir) / 'checkpoints'
        if not self.folder.is_dir():
            raise FileNotFoundError('Checkpoint directory not found: ' + str(self.folder))

    def records(self):
        records = []
        for path in self.folder.glob('step_*.json'):
            try:
                info = json.loads(path.read_text(encoding='utf-8'))
                assert isinstance(info['step'], int) and info['step'] >= 0
                assert isinstance(info['created_ns'], int)
                records.append((path, info))
            except Exception as error:
                print('Ignoring invalid manifest:', path.name, str(error))
        return sorted(records, key=lambda row: (row[1]['step'], row[1]['created_ns']), reverse=True)

    def load(self, which='latest'):
        assert which in ('latest', 'best')
        records = self.records()
        if which == 'best':
            records = sorted([row for row in records if row[1].get('validation_bpc') is not None], key=lambda row: row[1]['validation_bpc'])
            if not records:
                print('No evaluated best snapshot yet; using latest.')
                return self.load('latest')
        for path, info in records:
            try:
                state, _ = read_payload(path)
                required = {'model', 'optimizer', 'scheduler', 'scaler', 'rng', 'batch_rng', 'cfg', 'vocab', 'data_fingerprint', 'step', 'total_steps', 'format_version'}
                if not required.issubset(state) or state['format_version'] != FORMAT_VERSION:
                    raise ValueError('Incomplete or unsupported training state.')
                if state['step'] != info['step']:
                    raise ValueError('Checkpoint/manifest step mismatch.')
                print(f"Loaded {which}: update {state['step']:,} — {path.with_suffix('.pt')}")
                return state
            except Exception as error:
                print('Recovery: skipping', path.name, '—', str(error))
        if list(self.folder.glob('step_*.json')):
            raise RuntimeError('Existing checkpoints could not be loaded. Repair the folder or choose a NEW RUN_NAME; refusing to restart silently.')
        print('No complete checkpoint found. This is a fresh run.')
        return None

def decode_tokens(ids, vocab, mask_id):
    return ''.join('▮' if int(i) == mask_id else vocab[int(i)] for i in ids)

class TimeEmbedding(nn.Module):
    def __init__(self, width):
        super().__init__()
        frequencies = torch.exp(-math.log(10000) * torch.arange(width // 2) / (width // 2 - 1))
        self.register_buffer('frequencies', frequencies, persistent=False)
        self.mlp = nn.Sequential(nn.Linear(width, width), nn.SiLU(), nn.Linear(width, width))

    def forward(self, t):
        angles = t.float()[:, None] * self.frequencies[None, :]
        return self.mlp(torch.cat([angles.sin(), angles.cos()], dim=-1))

class TransformerBlock(nn.Module):
    def __init__(self, width, heads, ff_multiplier, dropout):
        super().__init__()
        self.heads, self.dropout = heads, dropout
        self.norm1 = nn.LayerNorm(width)
        self.qkv = nn.Linear(width, width * 3)
        self.attn_out = nn.Linear(width, width)
        self.norm2 = nn.LayerNorm(width)
        self.ff_in = nn.Linear(width, width * ff_multiplier)
        self.ff_out = nn.Linear(width * ff_multiplier, width)

    def forward(self, x):
        batch, length, width = x.shape
        qkv = self.qkv(self.norm1(x)).reshape(batch, length, 3, self.heads, width // self.heads)
        q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(0)
        attention = F.scaled_dot_product_attention(
            q, k, v, dropout_p=self.dropout if self.training else 0.0, is_causal=False)
        attention = attention.transpose(1, 2).reshape(batch, length, width)
        x = x + F.dropout(self.attn_out(attention), p=self.dropout, training=self.training)
        hidden = F.gelu(self.ff_in(self.norm2(x)))
        hidden = F.dropout(hidden, p=self.dropout, training=self.training)
        return x + F.dropout(self.ff_out(hidden), p=self.dropout, training=self.training)

class CharacterD3PM(nn.Module):
    def __init__(self, clean_vocab_size, config):
        super().__init__()
        width = config['width']
        assert width >= 4 and width % 2 == 0 and width % config['heads'] == 0
        self.seq_len = config['seq_len']
        self.mask_id = clean_vocab_size
        self.token = nn.Embedding(clean_vocab_size + 1, width)
        self.position = nn.Embedding(self.seq_len, width)
        self.time = TimeEmbedding(width)
        self.dropout = config['dropout']
        self.blocks = nn.ModuleList([
            TransformerBlock(width, config['heads'], config['ff_multiplier'], config['dropout'])
            for _ in range(config['layers'])])
        self.norm = nn.LayerNorm(width)
        self.head = nn.Linear(width, clean_vocab_size)
        self.apply(self._initialize)
        # Keep the initial residual stream scale stable as layers accumulate.
        for block in self.blocks:
            nn.init.normal_(block.attn_out.weight, std=0.02 / math.sqrt(2 * config['layers']))
            nn.init.normal_(block.ff_out.weight, std=0.02 / math.sqrt(2 * config['layers']))

    @staticmethod
    def _initialize(module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, std=0.02)
        if isinstance(module, nn.Linear) and module.bias is not None:
            nn.init.zeros_(module.bias)

    def forward(self, x_t, t):
        assert x_t.ndim == 2 and x_t.shape[1] <= self.seq_len
        positions = torch.arange(x_t.shape[1], device=x_t.device)
        x = self.token(x_t) + self.position(positions)[None] + self.time(t)[:, None]
        x = F.dropout(x, p=self.dropout, training=self.training)
        for block in self.blocks:
            x = block(x)
        return self.head(self.norm(x))

@torch.no_grad()
def reverse_step(model, xt, t, generator, temperature=1.0):
    assert t >= 1 and temperature > 0
    time_ids = torch.full((len(xt),), t, device=xt.device, dtype=torch.long)
    logits = model(xt, time_ids).float()
    unmask = xt.eq(model.mask_id) & (torch.rand(xt.shape, device=xt.device, generator=generator) < 1.0 / t)
    result = xt.clone()
    if unmask.any():
        probs = (logits[unmask] / temperature).softmax(dim=-1)
        result[unmask] = torch.multinomial(probs, 1, generator=generator).squeeze(-1)
    return result

@torch.no_grad()
def sample(model, config, target_device, batch_size=1, seed=123, temperature=1.0,
           template=None, snapshots=10):
    if temperature <= 0 or snapshots < 1:
        raise ValueError('temperature must be positive and snapshots must be at least 1.')
    generator = torch.Generator(device=target_device).manual_seed(seed)
    was_training = model.training
    model.eval()
    try:
        if template is None:
            canvas = torch.full((batch_size, config['seq_len']), model.mask_id,
                                dtype=torch.long, device=target_device)
        else:
            canvas = template.to(target_device).clone()
            if canvas.ndim == 1:
                canvas = canvas.unsqueeze(0)
            if canvas.ndim != 2 or not 0 < canvas.shape[1] <= config['seq_len']:
                raise ValueError('Template must contain between 1 and seq_len positions.')
            if ((canvas < 0) | (canvas > model.mask_id)).any():
                raise ValueError('Template contains an invalid token.')
        fixed = canvas.ne(model.mask_id)
        fixed_values = canvas.clone()
        frames = [(config['timesteps'], canvas.cpu().clone())]
        snapshot_steps = set(np.linspace(0, config['timesteps'], snapshots + 1, dtype=int).tolist())
        for t in range(config['timesteps'], 0, -1):
            with torch.autocast(device_type=target_device.type, dtype=torch.float16,
                                enabled=target_device.type == 'cuda'):
                canvas = reverse_step(model, canvas, t, generator, temperature)
            canvas[fixed] = fixed_values[fixed]
            if t - 1 in snapshot_steps:
                frames.append((t - 1, canvas.cpu().clone()))
        assert not canvas.eq(model.mask_id).any(), 'Reverse process left unresolved masks.'
        return canvas.cpu(), frames
    finally:
        model.train(was_training)


def load_model_for_generation(checkpoint_store, target_device, which='best'):
    """Restore architecture, vocabulary and weights without corpus or trainer setup."""
    state = checkpoint_store.load(which)
    if state is None:
        raise FileNotFoundError('No saved model is available.')
    # Constructor randomness must not affect the caller's CPU random stream.
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(0)
        loaded_model = CharacterD3PM(len(state['vocab']), state['cfg'])
    loaded_model.load_state_dict(state['model'], strict=True)
    loaded_model.to(target_device).eval()
    return loaded_model, state['cfg'], state['vocab'], state['step']
