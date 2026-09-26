"""Configuration: every hyperparameter of the method, with its default.

Any field can be overridden from the command line with `--set section.field=value` (the value is parsed as YAML),
e.g. `--set train.epochs=100 --set model.hidden=192`.
"""
from __future__ import annotations

import dataclasses as dc
import hashlib
import json
from typing import Sequence

import yaml

TOKENIZER_VERSION = 1


@dc.dataclass
class TokenizerCfg:
    z_dim: int = 64                 # width of the feature block F and of the vocabulary block
    rank_tol: float = 1e-6          # singular values <= rank_tol * max lie outside the numerical rank
    beta_order: int = 4             # C: Beta wavelet kernels beta_0..beta_C
    log_delta_eps: float = 1e-12    # log(delta + eps) guards delta = 0
    rwpe_steps: int = 8             # random-walk return profile length
    rwpe_probes: int = 64           # Rademacher probes for the k >= 3 return probabilities
    rwpe_seed: int = 0
    device: str = "cuda"            # falls back to CPU when CUDA is unavailable


@dc.dataclass
class ModelCfg:
    hidden: int = 128
    blocks: int = 3
    heads: int = 4
    cheb_order: int = 8             # K: Chebyshev order of the spectral experts (>= beta_order)
    dropout: float = 0.1
    router_hidden: int = 32
    ffn_mult: int = 4
    tau_start: float = 1.0          # router softmax temperature, annealed linearly ...
    tau_end: float = 0.5            # ... to this value over training
    warmup_epochs: int = 5          # epochs with the band warm-up KL term on the router
    warmup_weight: float = 0.1
    attn_neighbor_cap: int = 64     # neighbours per node sampled for attention while training
    attn_chunk_edges: int = 4_000_000
    attn_eval_max_edges: int = 100_000_000   # at evaluation every neighbour is used unless the graph is larger
    attn_bf16: bool = True


@dc.dataclass
class TrainCfg:
    epochs: int = 200
    lr: float = 1e-3
    weight_decay: float = 0.0
    grad_clip: float = 1.0
    source_val_frac: float = 0.1    # stratified hold-out of every source graph (model selection for "best")
    seeds: int = 1


@dc.dataclass
class Config:
    tokenizer: TokenizerCfg = dc.field(default_factory=TokenizerCfg)
    model: ModelCfg = dc.field(default_factory=ModelCfg)
    train: TrainCfg = dc.field(default_factory=TrainCfg)

    def hash(self, section: str) -> str:
        d = {k: v for k, v in dc.asdict(getattr(self, section)).items() if k != "device"}
        d["_version"] = TOKENIZER_VERSION
        return hashlib.sha1(json.dumps(d, sort_keys=True).encode()).hexdigest()[:10]

    def validate(self) -> "Config":
        if self.model.cheb_order < self.tokenizer.beta_order:
            raise ValueError("model.cheb_order must be >= tokenizer.beta_order")
        if self.model.hidden % self.model.heads:
            raise ValueError("model.hidden must be divisible by model.heads")
        return self


def load_config(overrides: Sequence[str] = ()) -> Config:
    cfg = Config()
    for o in overrides:
        key, sep, val = o.partition("=")
        if not sep:
            raise ValueError(f"override {o!r} is not KEY=VALUE")
        sec, _, field = key.partition(".")
        obj = getattr(cfg, sec, None)
        if obj is None or not hasattr(obj, field):
            raise KeyError(f"unknown config key {key}")
        setattr(obj, field, yaml.safe_load(val))
    return cfg.validate()
