"""Experiment 2: chapter-routed vs specialty-routed MoE (and stacked + parent conditioning) on CodiEsp.

Every variant uses the same frozen Bio_ClinicalBERT note embedding and predicts two code levels:
3-char ICD category ("cat") and full code ("full"). Only the routing / conditioning differs.

  flat                 one MLP (no experts)
  flat_parent          flat + child logits conditioned on parent-category logits (one model, trained jointly)
  separate_parent      two separate models: model 1 predicts categories; model 2 predicts full codes, conditioned
                       on model 1's category logits. Model 1's logits on the training notes are more confident
                       than on unseen notes, so model 2 trains on cleaner parents than it sees at test time.
  moe_learned          unsupervised gate, load-balancing loss
  moe_chapter          gate trained to predict the note's ICD chapters (chapter unknown at test time)
  moe_specialty        gate = pseudo-specialty probabilities (like MIMIC's services table, known at test time)
  moe_specialty_noMED  same, with the general-medicine (MED) expert removed
  stacked              specialty experts -> chapter experts -> FiLM(specialty, chapter) on the full-code
                       level + parent conditioning
  stacked_noparent     stacked without parent conditioning
  stacked_nofilm       stacked without FiLM
"""
import argparse
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from common import OUT

HIDDEN = 128
VARIANTS = ["flat", "flat_parent", "separate_parent", "moe_learned", "moe_chapter", "moe_specialty", "moe_specialty_noMED",
            "stacked", "stacked_noparent", "stacked_nofilm"]


def experts(n, d_in, d_out):
    return nn.ModuleList(nn.Sequential(nn.Linear(d_in, d_out), nn.ReLU()) for _ in range(n))


def mix(expert_list, gate, x):
    return torch.einsum("be,beh->bh", gate, torch.stack([e(x) for e in expert_list], 1))


class Model(nn.Module):
    def __init__(self, variant, d_in, n_cat, n_full, parent, n_chapters, n_specialties):
        super().__init__()
        self.v = variant
        self.drop = nn.Dropout(0.2)
        self.register_buffer("parent", torch.as_tensor(parent))
        stacked = variant.startswith("stacked")
        if variant.startswith("flat"):
            self.body = nn.Sequential(nn.Linear(d_in, 4 * HIDDEN), nn.ReLU(), nn.Linear(4 * HIDDEN, HIDDEN), nn.ReLU())
        elif variant == "moe_learned":
            self.gate, self.ex = nn.Linear(d_in, 8), experts(8, d_in, HIDDEN)
        elif variant == "moe_chapter":
            self.gate, self.ex = nn.Linear(d_in, n_chapters), experts(n_chapters, d_in, HIDDEN)
        elif variant.startswith("moe_specialty"):
            self.ex = experts(n_specialties, d_in, HIDDEN)
        if stacked:
            self.spec_ex = experts(n_specialties, d_in, HIDDEN)
            self.chap_gate = nn.Linear(d_in + HIDDEN, n_chapters)
            self.chap_ex = experts(n_chapters, HIDDEN, HIDDEN)
        self.use_film = stacked and variant != "stacked_nofilm"
        self.use_parent = variant in ("flat_parent", "stacked", "stacked_nofilm")
        if self.use_film:
            self.film = nn.Linear(n_specialties + n_chapters, 2 * HIDDEN)
            nn.init.zeros_(self.film.weight), nn.init.zeros_(self.film.bias)
        if self.use_parent:
            self.parent_w = nn.Parameter(torch.full((n_full,), 0.5))
        self.cat_head, self.full_head = nn.Linear(HIDDEN, n_cat), nn.Linear(HIDDEN, n_full)

    def forward(self, x, spec, parent_logits=None):
        x = self.drop(x)
        gate = None
        if self.v.startswith("flat"):
            z = self.body(x)
        elif self.v.startswith("moe_specialty"):
            gate = spec
            z = mix(self.ex, gate, x)
        elif self.v.startswith("moe"):
            gate = F.softmax(self.gate(x), -1)
            z = mix(self.ex, gate, x)
        else:
            z_s = mix(self.spec_ex, spec, x)
            gate = F.softmax(self.chap_gate(torch.cat([x, z_s], -1)), -1)
            z = z_s + mix(self.chap_ex, gate, z_s)
        cat = self.cat_head(z)
        z_full = z
        if self.use_film:
            gamma, beta = self.film(torch.cat([spec, gate], -1)).chunk(2, -1)
            z_full = z * (1 + gamma) + beta
        full = self.full_head(z_full)
        if self.use_parent:
            parents = cat if parent_logits is None else parent_logits
            full = full + self.parent_w * parents[:, self.parent]
        return cat, full, gate


def best_threshold_f1(probs, y):
    """Micro-F1 maximised over one global threshold. Returns (threshold, micro_f1)."""
    best = (0.5, 0.0)
    for t in np.arange(0.05, 0.95, 0.025):
        p = probs >= t
        tp = (p & (y == 1)).sum()
        f1 = 2 * tp / max(p.sum() + y.sum(), 1)
        best = max(best, (t, f1), key=lambda b: b[1])
    return best


def f1_scores(probs, y, t):
    p, y = probs >= t, y == 1
    tp, fp, fn = (p & y).sum(0), (p & ~y).sum(0), (~p & y).sum(0)
    micro = 2 * tp.sum() / max(2 * tp.sum() + fp.sum() + fn.sum(), 1)
    present = y.sum(0) > 0  # macro over labels with >=1 positive in this split
    macro = (2 * tp / np.maximum(2 * tp + fp + fn, 1))[present].mean()
    return micro, macro


def specialty_inputs(variant, data):
    """Per-split specialty routing probabilities; the noMED variant drops MED and renormalises."""
    spec = {s: torch.as_tensor(data[f"{s}_specialty"]) for s in ["train", "dev", "test"]}
    if variant == "moe_specialty_noMED":
        keep = torch.as_tensor([s != "MED" for s in data["services"]])
        spec = {k: v[:, keep] / v[:, keep].sum(1, keepdim=True) for k, v in spec.items()}
    return spec


def compute_loss(variant, cat, full, gate, y_cat, y_full, y_chapter):
    loss = F.binary_cross_entropy_with_logits(cat, y_cat) + F.binary_cross_entropy_with_logits(full, y_full)
    if variant == "moe_chapter" or variant.startswith("stacked"):
        target = y_chapter / y_chapter.sum(1, keepdim=True).clamp(min=1)
        loss = loss + 0.1 * -(target * torch.log(gate + 1e-8)).sum(1).mean()
    if variant == "moe_learned":
        loss = loss + 0.01 * gate.shape[1] * (gate.mean(0) ** 2).sum()
    return loss


def evaluate(variant, data, dev_probs, test_probs, test_gate, seed, params):
    """Tunes one threshold per level on dev, reports test F1 and routing stats."""
    tc = best_threshold_f1(dev_probs[0], data["dev_cat"])[0]
    tf = best_threshold_f1(dev_probs[1], data["dev_full"])[0]
    row = {"variant": variant, "seed": seed, "params": params}
    row["cat_micro"], row["cat_macro"] = f1_scores(test_probs[0], data["test_cat"], tc)
    row["full_micro"], row["full_macro"] = f1_scores(test_probs[1], data["test_full"], tf)
    if test_gate is not None:
        g = test_gate
        row["gate_entropy"] = float(-(g * np.log(g + 1e-8)).sum(1).mean() / np.log(g.shape[1]))
        row["experts_used_80pct"] = float(np.mean((np.sort(g, 1)[:, ::-1].cumsum(1) < 0.8).sum(1) + 1))
    return row


def fit(model, train_step, dev_score, epochs=400):
    """Full-batch AdamW; keeps the weights with the best dev score (checked every 10 epochs)."""
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)
    best, best_state = -1, None
    for epoch in range(epochs):
        loss = train_step()
        opt.zero_grad(), loss.backward(), opt.step()
        if epoch % 10 == 9:
            model.eval()
            with torch.no_grad():
                score = dev_score()
            model.train()
            if score > best:
                best, best_state = score, {k: v.clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    model.eval()


def run_separate(data, seed, epochs=400):
    """Two independently trained models: categories first, then full codes conditioned on them."""
    torch.manual_seed(seed), np.random.seed(seed)
    t = {k: torch.as_tensor(v) for k, v in data.items() if v.dtype.kind in "fiub"}
    spec = specialty_inputs("flat", data)
    dims = (t["train_x"].shape[1], t["train_cat"].shape[1], t["train_full"].shape[1], data["parent"],
            t["train_chapter"].shape[1], spec["train"].shape[1])
    first, second = Model("flat", *dims), Model("flat_parent", *dims)

    fit(first, lambda: F.binary_cross_entropy_with_logits(first(t["train_x"], spec["train"])[0], t["train_cat"]),
        lambda: best_threshold_f1(torch.sigmoid(first(t["dev_x"], spec["dev"])[0]).numpy(), data["dev_cat"])[1], epochs)
    with torch.no_grad():
        cat_logits = {s: first(t[f"{s}_x"], spec[s])[0] for s in ["train", "dev", "test"]}

    fit(second, lambda: F.binary_cross_entropy_with_logits(
            second(t["train_x"], spec["train"], cat_logits["train"])[1], t["train_full"]),
        lambda: best_threshold_f1(torch.sigmoid(second(t["dev_x"], spec["dev"], cat_logits["dev"])[1]).numpy(),
                                  data["dev_full"])[1], epochs)
    with torch.no_grad():
        probs = {s: (torch.sigmoid(cat_logits[s]).numpy(),
                     torch.sigmoid(second(t[f"{s}_x"], spec[s], cat_logits[s])[1]).numpy(), None) for s in ["dev", "test"]}
    params = sum(p.numel() for p in first.parameters()) + sum(p.numel() for p in second.parameters())
    return evaluate("separate_parent", data, probs["dev"], probs["test"], None, seed, params)


def run(variant, data, seed, epochs=400):
    torch.manual_seed(seed), np.random.seed(seed)
    t = {k: torch.as_tensor(v) for k, v in data.items() if v.dtype.kind in "fiub"}
    spec = specialty_inputs(variant, data)
    model = Model(variant, t["train_x"].shape[1], t["train_cat"].shape[1], t["train_full"].shape[1],
                  data["parent"], t["train_chapter"].shape[1], spec["train"].shape[1])

    def predict(split):
        with torch.no_grad():
            cat, full, gate = model(t[f"{split}_x"], spec[split])
        return torch.sigmoid(cat).numpy(), torch.sigmoid(full).numpy(), None if gate is None else gate.numpy()

    def train_step():
        cat, full, gate = model(t["train_x"], spec["train"])
        return compute_loss(variant, cat, full, gate, t["train_cat"], t["train_full"], t["train_chapter"])

    def dev_score():
        pc, pf, _ = predict("dev")
        return best_threshold_f1(pc, data["dev_cat"])[1] + best_threshold_f1(pf, data["dev_full"])[1]

    fit(model, train_step, dev_score, epochs)
    dev, test = predict("dev"), predict("test")
    return evaluate(variant, data, dev, test, test[2], seed, sum(p.numel() for p in model.parameters()))


def report(df, name, title):
    """Writes <name>.md (mean ± std table) and <name>.png from per-seed rows."""
    metrics = ["cat_micro", "cat_macro", "full_micro", "full_macro"]
    agg = df.groupby("variant", sort=False)[metrics].agg(["mean", "std"])
    extra = df.groupby("variant", sort=False)[["params", "gate_entropy", "experts_used_80pct"]].mean()
    table = pd.DataFrame({m: [f"{agg.loc[v, (m, 'mean')]:.3f} ± {agg.loc[v, (m, 'std')]:.3f}" for v in agg.index]
                          for m in metrics}, index=agg.index)
    table["params"] = extra.params.map(lambda p: f"{p / 1e3:.0f}k")
    table["gate entropy"] = extra.gate_entropy.map(lambda e: "" if pd.isna(e) else f"{e:.2f}")
    table["experts for 80% weight"] = extra.experts_used_80pct.map(lambda e: "" if pd.isna(e) else f"{e:.1f}")
    md = table.to_markdown()
    (OUT / f"{name}.md").write_text(md + "\n")
    print(md)

    n_seeds = df.seed.nunique()
    fig, axes = plt.subplots(1, 2, figsize=(12, 0.5 * len(agg) + 2), sharey=False)
    for ax, level in zip(axes, ["cat", "full"]):
        for i, kind in enumerate(["micro", "macro"]):
            m = agg[(f"{level}_{kind}", "mean")]
            s = agg[(f"{level}_{kind}", "std")]
            ax.barh(np.arange(len(m)) + i * 0.4, m, 0.4, xerr=s, label=f"{kind}-F1", capsize=2)
        ax.set_yticks(np.arange(len(agg)) + 0.2, agg.index)
        ax.invert_yaxis()
        ax.set_title({"cat": "3-char category", "full": "Full ICD-10 code"}[level])
        ax.set_xlabel(f"test F1 (mean ± std, {n_seeds} seeds)")
    fig.legend(*axes[0].get_legend_handles_labels(), loc="lower center", ncol=2)
    fig.suptitle(title)
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    fig.savefig(OUT / f"{name}.png", dpi=150, bbox_inches="tight")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--plot-only", action="store_true", help="redraw from the saved per-seed CSV")
    args = ap.parse_args()
    title = "CodiEsp (English) pilot, frozen encoder: routing strategy vs ICD coding F1"
    if args.plot_only:
        return report(pd.read_csv(OUT / "moe_results_by_seed.csv"), "moe_results", title)
    data = dict(np.load(OUT / "codiesp_features.npz", allow_pickle=True))
    mu, sd = data["train_x"].mean(0), data["train_x"].std(0) + 1e-6
    for s in ["train", "dev", "test"]:
        data[f"{s}_x"] = ((data[f"{s}_x"] - mu) / sd).astype(np.float32)
    torch.set_num_threads(8)

    rows = []
    for v in VARIANTS:
        for s in range(args.seeds):
            rows.append(run_separate(data, s) if v == "separate_parent" else run(v, data, s))
        r = pd.DataFrame(rows[-args.seeds:])
        print(f"{v:22s} cat micro {r.cat_micro.mean():.3f}  full micro {r.full_micro.mean():.3f}  "
              f"full macro {r.full_macro.mean():.3f}")
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "moe_results_by_seed.csv", index=False)
    report(df, "moe_results", title)


if __name__ == "__main__":
    main()
