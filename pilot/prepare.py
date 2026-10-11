"""Builds inputs for the MoE experiment:
1. Pseudo-specialty for each CodiEsp note, from a TF-IDF classifier trained on MTSamples.
2. Frozen Bio_ClinicalBERT embeddings for each CodiEsp note (512-token chunks, mean-pooled).
3. Label matrices: 3-char ICD categories, full codes, chapters.
"""
import json
import numpy as np
import pandas as pd
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from transformers import AutoModel, AutoTokenizer

from common import CODIESP, DATA, OUT, SERVICES, MTSAMPLES_TO_SERVICE, CHAPTER_NAMES, icd10_chapter

ENCODER = "emilyalsentzer/Bio_ClinicalBERT"
MIN_COUNT = 3  # drop codes seen <3 times in train; the rare-code tail needs MIMIC-scale data
SPLITS = ["train", "dev", "test"]


def load_codiesp():
    docs = {}
    for s in SPLITS:
        labels = pd.read_csv(CODIESP / s / f"{s}D.tsv", sep="\t", header=None, names=["doc", "code"])
        labels = labels.groupby("doc").code.apply(lambda c: sorted(set(c.str.lower())))
        rows = []
        for path in sorted((CODIESP / s / "text_files_en").glob("*.txt")):
            rows.append({"doc": path.stem, "text": path.read_text(), "codes": labels.get(path.stem, [])})
        docs[s] = pd.DataFrame(rows)
    return docs


def pseudo_specialty(docs):
    mt = pd.read_csv(DATA / "mtsamples.csv").dropna(subset=["transcription"])
    mt["service"] = mt.medical_specialty.str.strip().map(MTSAMPLES_TO_SERVICE)
    mt = mt.dropna(subset=["service"])
    # MTSamples repeats one note under several specialties (mostly "X" + "Surgery"), which leaks
    # across splits with conflicting labels. Keep one row per text, preferring the rarest label,
    # so specific services beat the generic SURG.
    freq = mt.service.value_counts()
    mt = (mt.assign(freq=mt.service.map(freq)).sort_values("freq")
            .drop_duplicates("transcription").drop(columns="freq"))
    x_tr, x_te, y_tr, y_te = train_test_split(mt.transcription, mt.service, test_size=0.2,
                                              stratify=mt.service, random_state=0)
    vec = TfidfVectorizer(sublinear_tf=True, min_df=2, ngram_range=(1, 2), max_features=50000, stop_words="english")
    clf = LogisticRegression(max_iter=2000, C=5, class_weight="balanced")
    clf.fit(vec.fit_transform(x_tr), y_tr)
    pred = clf.predict(vec.transform(x_te))
    stats = {"mtsamples_notes": len(mt), "heldout_accuracy": accuracy_score(y_te, pred),
             "heldout_macro_f1": f1_score(y_te, pred, average="macro"),
             "class_counts": mt.service.value_counts().to_dict()}
    assert list(clf.classes_) == SERVICES
    for s in SPLITS:
        docs[s]["specialty_probs"] = list(clf.predict_proba(vec.transform(docs[s].text)))
    dist = pd.Series(np.concatenate([docs[s].specialty_probs.apply(np.argmax) for s in SPLITS]))
    stats["codiesp_argmax_distribution"] = {SERVICES[i]: int(n) for i, n in dist.value_counts().items()}
    return stats


@torch.no_grad()
def embed(texts, device):
    tok = AutoTokenizer.from_pretrained(ENCODER)
    model = AutoModel.from_pretrained(ENCODER).to(device).eval()
    out = []
    for i, text in enumerate(texts):
        ids = tok(text, add_special_tokens=False)["input_ids"]
        chunks = [ids[j:j + 510] for j in range(0, max(len(ids), 1), 510)]
        batch = tok.pad({"input_ids": [[tok.cls_token_id] + c + [tok.sep_token_id] for c in chunks]},
                        return_tensors="pt").to(device)
        hidden = model(**batch).last_hidden_state
        mask = batch["attention_mask"].unsqueeze(-1)
        chunk_vecs = (hidden * mask).sum(1) / mask.sum(1)
        out.append(chunk_vecs.mean(0).cpu().numpy())
        if i % 200 == 0:
            print(f"  embedded {i}/{len(texts)}")
    return np.stack(out)


def label_matrix(code_lists, vocab):
    index = {c: i for i, c in enumerate(vocab)}
    y = np.zeros((len(code_lists), len(vocab)), dtype=np.float32)
    for r, codes in enumerate(code_lists):
        for c in codes:
            if c in index:
                y[r, index[c]] = 1
    return y


def main():
    docs = load_codiesp()
    stats = pseudo_specialty(docs)
    print(json.dumps(stats, indent=1))

    train_codes = docs["train"].codes
    full_counts = pd.Series([c for cs in train_codes for c in cs]).value_counts()
    cat_counts = pd.Series([c for cs in train_codes for c in {x.split(".")[0] for x in cs}]).value_counts()
    full_vocab = sorted(full_counts[full_counts >= MIN_COUNT].index)
    cat_vocab = sorted(cat_counts[cat_counts >= MIN_COUNT].index)
    # parent index of each full code in the category vocab (-1 if its category was filtered out)
    parent = [cat_vocab.index(c.split(".")[0]) if c.split(".")[0] in cat_vocab else -1 for c in full_vocab]
    stats.update(n_full_codes=len(full_vocab), n_categories=len(cat_vocab),
                 full_codes_with_parent=int(sum(p >= 0 for p in parent)))

    device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
    arrays = {"full_vocab": np.array(full_vocab), "cat_vocab": np.array(cat_vocab),
              "parent": np.array(parent), "services": np.array(SERVICES), "chapters": np.array(CHAPTER_NAMES)}
    for s in SPLITS:
        d = docs[s]
        print(f"{s}: embedding {len(d)} notes on {device}")
        arrays[f"{s}_x"] = embed(d.text.tolist(), device)
        arrays[f"{s}_full"] = label_matrix(d.codes, full_vocab)
        arrays[f"{s}_cat"] = label_matrix(d.codes.apply(lambda cs: {c.split('.')[0] for c in cs}), cat_vocab)
        arrays[f"{s}_chapter"] = label_matrix(d.codes.apply(lambda cs: {icd10_chapter(c) for c in cs}), CHAPTER_NAMES)
        arrays[f"{s}_specialty"] = np.stack(d.specialty_probs).astype(np.float32)
    np.savez_compressed(OUT / "codiesp_features.npz", **arrays)
    (OUT / "prepare_stats.json").write_text(json.dumps(stats, indent=1))
    print(json.dumps({k: v for k, v in stats.items() if k.startswith("n_") or k.startswith("full")}, indent=1))


if __name__ == "__main__":
    main()
