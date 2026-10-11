"""Experiment 2b (GPU): same routing variants as run_moe.py, but Bio_ClinicalBERT is fine-tuned end to end
instead of frozen. Meant for a free Kaggle/Colab GPU; about 10 min per run on a T4.
Requires prepare.py to have run (labels + pseudo-specialty come from codiesp_features.npz).
"""
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from transformers import AutoModel, AutoTokenizer
from common import OUT
from prepare import ENCODER, SPLITS, load_codiesp
from run_moe import Model, best_threshold_f1, compute_loss, evaluate, report, specialty_inputs

MAX_CHUNKS = 3  # 3 x 510 tokens covers ~99% of CodiEsp notes (99th percentile ~1.2k tokens)
DEFAULT_VARIANTS = ["flat", "flat_parent", "moe_chapter", "moe_specialty", "stacked"]


class Coder(nn.Module):
    def __init__(self, variant, data, n_specialties):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(ENCODER)
        d = self.encoder.config.hidden_size
        self.norm = nn.LayerNorm(d)
        self.head = Model(variant, d, data["train_cat"].shape[1], data["train_full"].shape[1], data["parent"],
                          data["train_chapter"].shape[1], n_specialties)

    def forward(self, chunks, doc_index, n_docs, spec):
        h = self.encoder(**chunks).last_hidden_state
        mask = chunks["attention_mask"].unsqueeze(-1)
        chunk_vecs = ((h * mask).sum(1) / mask.sum(1)).float()
        doc = torch.zeros(n_docs, h.shape[-1], device=h.device).index_add_(0, doc_index, chunk_vecs)
        doc = doc / torch.bincount(doc_index, minlength=n_docs).unsqueeze(1)
        # heads stay fp32: the gate loss takes log(gate), which underflows in fp16
        with torch.autocast(h.device.type, enabled=False):
            return self.head(self.norm(doc), spec)


def tokenize(tok, texts):
    out = []
    for text in texts:
        ids = tok(text, add_special_tokens=False)["input_ids"][: 510 * MAX_CHUNKS]
        out.append([[tok.cls_token_id] + ids[j:j + 510] + [tok.sep_token_id] for j in range(0, max(len(ids), 1), 510)])
    return out


def make_batch(tok, doc_tokens, idx, device):
    chunks, doc_index = [], []
    for k, i in enumerate(idx):
        chunks += doc_tokens[i]
        doc_index += [k] * len(doc_tokens[i])
    return tok.pad({"input_ids": chunks}, return_tensors="pt").to(device), torch.tensor(doc_index, device=device)


def run(variant, data, tokens, tok, seed, epochs, batch_size, device):
    torch.manual_seed(seed), np.random.seed(seed)
    spec = {k: v.to(device) for k, v in specialty_inputs(variant, data).items()}
    model = Coder(variant, data, spec["train"].shape[1]).to(device)
    head_params = list(model.norm.parameters()) + list(model.head.parameters())
    opt = torch.optim.AdamW([{"params": model.encoder.parameters(), "lr": 2e-5},
                             {"params": head_params, "lr": 1e-3}], weight_decay=1e-2)
    y = {k: torch.as_tensor(data[f"train_{k}"]).to(device) for k in ["cat", "full", "chapter"]}
    amp = device == "cuda"
    scaler = torch.amp.GradScaler(enabled=amp)

    def predict(split):
        model.eval()
        cats, fulls, gates = [], [], []
        with torch.no_grad(), torch.autocast(device, enabled=amp):
            for s in range(0, len(tokens[split]), 2 * batch_size):
                idx = list(range(s, min(s + 2 * batch_size, len(tokens[split]))))
                chunks, di = make_batch(tok, tokens[split], idx, device)
                cat, full, gate = model(chunks, di, len(idx), spec[split][idx])
                cats.append(torch.sigmoid(cat).cpu()), fulls.append(torch.sigmoid(full).cpu())
                gates.append(None if gate is None else gate.cpu())
        model.train()
        return torch.cat(cats).numpy(), torch.cat(fulls).numpy(), None if gates[0] is None else torch.cat(gates).numpy()

    best, best_state = -1, None
    for epoch in range(epochs):
        perm = np.random.permutation(len(tokens["train"]))
        for s in range(0, len(perm), batch_size):
            idx = perm[s:s + batch_size].tolist()
            chunks, di = make_batch(tok, tokens["train"], idx, device)
            with torch.autocast(device, enabled=amp):
                cat, full, gate = model(chunks, di, len(idx), spec["train"][idx])
            loss = compute_loss(variant, cat, full, gate, y["cat"][idx], y["full"][idx], y["chapter"][idx])
            opt.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
        pc, pf, _ = predict("dev")
        score = best_threshold_f1(pc, data["dev_cat"])[1] + best_threshold_f1(pf, data["dev_full"])[1]
        print(f"  {variant} seed {seed} epoch {epoch + 1}: loss {loss.item():.4f}  dev cat+full micro-F1 {score:.3f}")
        if score > best:
            best, best_state = score, {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    dev, test = predict("dev"), predict("test")
    return evaluate(variant, data, dev, test, test[2], seed, sum(p.numel() for p in head_params))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variants", nargs="+", default=DEFAULT_VARIANTS)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch-size", type=int, default=4, help="notes per step (each up to 3 chunks)")
    ap.add_argument("--smoke", action="store_true", help="16 notes per split, 1 epoch: checks the code runs")
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")

    data = dict(np.load(OUT / "codiesp_features.npz", allow_pickle=True))
    tok = AutoTokenizer.from_pretrained(ENCODER)
    docs = load_codiesp()  # same sorted order prepare.py used to build the npz
    tokens = {s: tokenize(tok, docs[s].text) for s in SPLITS}
    for s in SPLITS:
        assert len(tokens[s]) == len(data[f"{s}_cat"])
    name = "finetune_results"
    if args.smoke:
        name, args.epochs, args.seeds = "finetune_smoke", 1, 1
        for s in SPLITS:
            tokens[s] = tokens[s][:16]
            for k in ["x", "cat", "full", "chapter", "specialty"]:
                data[f"{s}_{k}"] = data[f"{s}_{k}"][:16]

    rows = []
    for v in args.variants:
        for seed in range(args.seeds):
            rows.append(run(v, data, tokens, tok, seed, args.epochs, args.batch_size, device))
            pd.DataFrame(rows).to_csv(OUT / f"{name}_by_seed.csv", index=False)  # survives a killed session
            print(rows[-1])
    report(pd.DataFrame(rows), name, "CodiEsp (English), fine-tuned Bio_ClinicalBERT: routing strategy vs ICD coding F1")


if __name__ == "__main__":
    main()
