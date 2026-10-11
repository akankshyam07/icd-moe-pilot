"""Experiment 1: how does MIMIC hospital service relate to ICD chapter? (MIMIC-IV Demo, 275 admissions)"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import mutual_info_score

from common import DATA, OUT, icd_chapter

d = DATA / "mimic-demo"
services = pd.read_csv(d / "services.csv.gz", parse_dates=["transfertime"])
dx = pd.read_csv(d / "diagnoses_icd.csv.gz")

# First service of each admission = the specialty label a router would see.
first = services.sort_values("transfertime").groupby("hadm_id").curr_service.first()
n_services = services.groupby("hadm_id").curr_service.nunique()
dx["chapter"] = [icd_chapter(c, v) for c, v in zip(dx.icd_code, dx.icd_version)]
dx = dx.dropna(subset=["chapter"])
dx = dx[dx.hadm_id.isin(first.index)]
chapters = dx.groupby("hadm_id").chapter.apply(set)
first = first.loc[chapters.index]

lines = [f"Admissions with service + diagnoses: {len(first)}"]
lines.append(f"Share whose first service is MED (general medicine): {(first == 'MED').mean():.1%}")
lines.append(f"Admissions that change service during stay: {(n_services.loc[first.index] > 1).mean():.1%}")
per_adm = chapters.apply(len)
lines.append(f"ICD chapters per admission: mean {per_adm.mean():.1f}, median {per_adm.median():.0f}, max {per_adm.max()}")

# Service x chapter: fraction of each service's admissions with >=1 code in that chapter.
long = chapters.explode().rename("chapter").reset_index().merge(first.rename("service").reset_index())
counts = first.value_counts()
keep = counts[counts >= 5].index
table = pd.crosstab(long.service, long.chapter).loc[keep]
frac = table.div(counts.loc[keep], axis=0)
frac = frac[frac.mean().sort_values(ascending=False).index]
frac.to_csv(OUT / "mimic_service_by_chapter.csv")

# Primary-diagnosis chapter (seq_num == 1) vs service: how much does service tell us?
primary = dx[dx.seq_num == 1].set_index("hadm_id").chapter
both = pd.concat([first.rename("service"), primary.rename("chapter")], axis=1).dropna()
mi = mutual_info_score(both.service, both.chapter)
h_chapter = mutual_info_score(both.chapter, both.chapter)
lines.append(f"Primary-diagnosis chapter uncertainty explained by first service (MI / H): {mi / h_chapter:.1%}")
top = both.groupby("service").chapter.agg(lambda s: s.value_counts(normalize=True).iloc[0])
lines.append("Top-chapter purity of primary diagnosis per service (n>=5): " +
             ", ".join(f"{s} {top[s]:.0%}" for s in keep))

fig, ax = plt.subplots(figsize=(11, 4.5))
im = ax.imshow(frac.values, cmap="Blues", vmin=0, vmax=1, aspect="auto")
ax.set_xticks(range(frac.shape[1]), frac.columns, rotation=60, ha="right")
ax.set_yticks(range(frac.shape[0]), [f"{s} (n={counts[s]})" for s in frac.index])
for i in range(frac.shape[0]):
    for j in range(frac.shape[1]):
        v = frac.values[i, j]
        if v >= 0.3:
            ax.text(j, i, f"{v:.0%}", ha="center", va="center", fontsize=7, color="white" if v > 0.6 else "black")
fig.colorbar(im, label="share of admissions with >=1 code in chapter")
ax.set_title("MIMIC-IV Demo: hospital service vs ICD chapter")
fig.tight_layout()
fig.savefig(OUT / "mimic_service_by_chapter.png", dpi=150)

report = "\n".join(lines)
(OUT / "mimic_demo_summary.txt").write_text(report + "\n")
print(report)
