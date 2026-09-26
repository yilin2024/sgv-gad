"""Train on labelled source graphs, then score unseen target graphs zero-shot.

    python -m sgv.train --data_root DATA --sources g1,g2 --targets g3,g4 [--seeds 3] [--out runs/demo] \
        [--set train.epochs=200 --set model.hidden=128 ...]

Protocol: nothing computed from a target graph reaches training, and target labels are read only to compute AUROC
and AUPRC. Every input is normalised per graph, from that graph's own unlabelled nodes. Each epoch takes one Adam
step per source graph. 10% of every source graph is held out (stratified, fixed per graph) and the epoch with the best
mean source-validation AUPRC gives the "best" weights; "last" is the final epoch.
"""
from __future__ import annotations

import argparse
import csv
import dataclasses as dc
import os
import time
import zlib
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as Fn
import yaml
from scipy.stats import rankdata
from sklearn.metrics import average_precision_score, roc_auc_score

from .config import Config, load_config
from .data import Graph, load_graph
from .graph_ops import GraphOps
from .model import Backbone, Neighbourhood, capped_neighbourhood
from .tokenizer import load_or_tokenize


# ------------------------------------------------------------------------------------------------ per-graph inputs
def rank_ext(X: np.ndarray) -> np.ndarray:
    """Per-graph ECDF rank r of every column as 2r - 1, with the two-sided extremeness |2r - 1| appended."""
    S = np.column_stack([2.0 * rankdata(X[:, j], method="average") / X.shape[0] - 1.0 for j in range(X.shape[1])])
    S = S.astype(np.float32)
    return np.c_[S, np.abs(S)]


def source_val_mask(name: str, y: np.ndarray, frac: float) -> np.ndarray:
    """Stratified hold-out, fixed per graph (independent of the training seed)."""
    rng = np.random.default_rng(zlib.crc32(name.encode()))
    val = np.zeros(len(y), bool)
    for c in (0, 1):
        idx = np.flatnonzero(y == c)
        val[rng.choice(idx, max(1, int(round(frac * len(idx)))), replace=False)] = True
    return val


@dc.dataclass
class GraphData:
    name: str
    ops: GraphOps
    x: torch.Tensor                 # [F || omega || joint], rank-normalised
    pos: torch.Tensor               # [zeta || log d], rank-normalised
    phi_t: torch.Tensor             # (n, C) router warm-up target
    y: np.ndarray
    yt: torch.Tensor
    crow: torch.Tensor
    col: torch.Tensor
    tr: Optional[torch.Tensor] = None
    va_np: Optional[np.ndarray] = None
    pw: Optional[torch.Tensor] = None
    nbr: Optional[Neighbourhood] = None
    _full: Optional[Neighbourhood] = None

    def resample(self, cap: int, gen: Optional[torch.Generator]) -> None:
        self.nbr = capped_neighbourhood(self.crow, self.col, cap, gen)

    def use_full(self, cap: int, max_edges: int) -> None:
        if self._full is None:
            self._full = capped_neighbourhood(self.crow, self.col, None if self.col.numel() <= max_edges else cap)
        self.nbr = self._full


def build_graph_data(g: Graph, tok: Dict[str, np.ndarray], cfg: Config, device: str, source: bool) -> GraphData:
    x = np.c_[rank_ext(tok["F"].astype(np.float64)), rank_ext(tok["omega"].astype(np.float64)),
              rank_ext(tok["joint"].astype(np.float64))]
    pos = rank_ext(np.c_[tok["zeta"], tok["logdeg"]].astype(np.float64))
    p = tok["phi"].astype(np.float64)
    s = p.sum(1, keepdims=True)
    C = p.shape[1]
    phi_t = np.where(s > 0, p / np.maximum(s, 1e-12), 1.0 / C)
    y = tok["y"].astype(np.int64)
    t = lambda a: torch.as_tensor(a, device=device)
    gd = GraphData(name=g.name, ops=GraphOps(g.A, device=device, dtype=torch.float32), x=t(x), pos=t(pos),
                   phi_t=torch.as_tensor(phi_t, dtype=torch.float32, device=device), y=y,
                   yt=t(y.astype(np.float32)), crow=t(g.A.indptr.astype(np.int64)), col=t(g.A.indices.astype(np.int64)))
    if source:
        va = source_val_mask(g.name, y, cfg.train.source_val_frac)
        tr = ~va
        gd.tr, gd.va_np = t(tr), va
        gd.pw = torch.tensor((y[tr] == 0).sum() / max((y[tr] == 1).sum(), 1), dtype=torch.float32, device=device)
    return gd


# ------------------------------------------------------------------------------------------------ training
def kappa_at(cfg: Config, t: int) -> float:
    mc, E = cfg.model, cfg.train.epochs
    return mc.tau_start + (mc.tau_end - mc.tau_start) * t / max(E - 1, 1)


def warmup_kl(alphas, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    p = target[mask].clamp_min(1e-12)
    return torch.stack([(p * (torch.log(p) - torch.log(a[mask].clamp_min(1e-12)))).sum(-1).mean()
                        for a in alphas]).mean()


@torch.no_grad()
def score_graph(model: nn.Module, g: GraphData, cfg: Config, kappa: float) -> np.ndarray:
    model.eval()
    g.use_full(cfg.model.attn_neighbor_cap, cfg.model.attn_eval_max_edges)
    a, _ = model(g, kappa)
    return a.float().cpu().numpy()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--data_root", required=True, help="directory holding one {name}.mat per graph")
    ap.add_argument("--sources", required=True, help="comma list of training graphs")
    ap.add_argument("--targets", required=True, help="comma list of zero-shot target graphs")
    ap.add_argument("--seeds", type=int, default=None, help="number of seeds (overrides train.seeds)")
    ap.add_argument("--out", default="runs/default", help="output directory")
    ap.add_argument("--cache_dir", default="cache", help="token cache directory ('' disables caching)")
    ap.add_argument("--set", action="append", default=[], metavar="KEY=VALUE", help="config override (repeatable)")
    a = ap.parse_args()
    cfg = load_config(a.set)
    if a.seeds is not None:
        cfg.train.seeds = a.seeds
    src = a.sources.split(",")
    tgt = [n for n in a.targets.split(",") if n not in src]
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(a.out, exist_ok=True)
    with open(os.path.join(a.out, "config.yaml"), "w") as f:
        yaml.safe_dump(dict(sources=src, targets=tgt, **dc.asdict(cfg)), f, sort_keys=False)

    graphs = {n: load_graph(a.data_root, n) for n in src + tgt}
    toks = {n: load_or_tokenize(graphs[n], cfg, a.cache_dir or None) for n in src + tgt}
    S = {n: build_graph_data(graphs[n], toks[n], cfg, dev, source=True) for n in src}
    tc, mc = cfg.train, cfg.model
    first = next(iter(S.values()))
    C = first.phi_t.shape[1]
    results = []
    print(f"[train] sources={src} targets={tgt} epochs={tc.epochs} seeds={tc.seeds} device={dev}", flush=True)
    for seed in range(tc.seeds):
        torch.manual_seed(seed)
        np.random.seed(seed)
        gen = torch.Generator(device=dev).manual_seed(seed)
        model = Backbone(mc, first.x.shape[1], first.pos.shape[1], C).to(dev)
        if seed == 0:
            print(f"[train] params = {sum(p.numel() for p in model.parameters()):,}", flush=True)
        opt = torch.optim.Adam(model.parameters(), lr=tc.lr, weight_decay=tc.weight_decay)
        best_val, best_ep, best_state = -1.0, -1, None
        for t in range(tc.epochs):
            t0 = time.time()
            kappa = kappa_at(cfg, t)
            losses = {}
            for n, g in S.items():
                model.train()
                g.resample(mc.attn_neighbor_cap, gen)
                out, alphas = model(g, kappa)
                loss = Fn.binary_cross_entropy_with_logits(out[g.tr], g.yt[g.tr], pos_weight=g.pw)
                losses[n] = float(loss)
                if t < mc.warmup_epochs:
                    loss = loss + mc.warmup_weight * warmup_kl(alphas, g.phi_t, g.tr)
                opt.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), tc.grad_clip)
                opt.step()
            vals = [average_precision_score(g.y[g.va_np], score_graph(model, g, cfg, kappa)[g.va_np])
                    for g in S.values()]
            if np.mean(vals) > best_val:
                best_val, best_ep = float(np.mean(vals)), t
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            if t % 10 == 0 or t == tc.epochs - 1:
                ltxt = " ".join(f"{n}={v:.3f}" for n, v in losses.items())
                print(f"[train] s{seed} ep{t:4d} kappa={kappa:.2f} loss {ltxt} | src-val AUPRC "
                      f"{np.mean(vals):.4f} (best {best_val:.4f} @ {best_ep}) {time.time() - t0:.1f}s", flush=True)
        kappa = kappa_at(cfg, max(tc.epochs - 1, 0))
        last_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        best_state = best_state or last_state
        for n in tgt:
            g = build_graph_data(graphs[n], toks[n], cfg, dev, source=False)
            for when, state in (("last", last_state), ("best", best_state)):
                model.load_state_dict(state)
                s = score_graph(model, g, cfg, kappa)
                results.append(dict(seed=seed, when=when, target=n, auroc=roc_auc_score(g.y, s),
                                    auprc=average_precision_score(g.y, s), best_epoch=best_ep))
            del g
            if dev == "cuda":
                torch.cuda.empty_cache()
        model.load_state_dict(last_state)

    with open(os.path.join(a.out, "results.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, list(results[0]))
        w.writeheader()
        w.writerows(results)
    print(f"\n[train] targets, mean over {tc.seeds} seed(s), x100")
    print(f"{'target':14s} {'last AUROC/AUPRC':>18s} {'best AUROC/AUPRC':>18s}")
    for n in tgt:
        cells = []
        for when in ("last", "best"):
            rr = [r for r in results if r["target"] == n and r["when"] == when]
            cells.append(f"{100 * np.mean([r['auroc'] for r in rr]):6.2f} / {100 * np.mean([r['auprc'] for r in rr]):5.2f}")
        print(f"{n:14s} {cells[0]:>18s} {cells[1]:>18s}")
    for when in ("last", "best"):
        rr = [r for r in results if r["when"] == when]
        per_seed = [np.mean([r["auroc"] for r in rr if r["seed"] == s]) for s in range(tc.seeds)]
        print(f"[train] {when:4s} mean AUROC {100 * np.mean(per_seed):.2f} +-{100 * np.std(per_seed):.2f}  "
              f"AUPRC {100 * np.mean([r['auprc'] for r in rr]):.2f}")
    print(f"[train] wrote {os.path.join(a.out, 'results.csv')}", flush=True)


if __name__ == "__main__":
    main()
