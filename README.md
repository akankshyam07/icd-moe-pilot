# ICD coding pilot: specialty vs. chapter routing

Pilot study for the VIP automated ICD coding project. It compares mixture-of-experts models whose experts
are split by **medical specialty** or by **ICD chapter**, along with parent-child conditioning and a stacked
specialty → chapter design. Everything runs on open data while we wait on the full MIMIC-IV notes.

**Report:** https://akankshyam07.github.io/icd-moe-pilot/

## What's here

| Path | Contents |
|---|---|
| `index.html` | The report (served by GitHub Pages) |
| `pilot/` | Code, results and a Kaggle notebook ([details](pilot/README.md)) |
| `pilot/out/` | Result tables and plots |
| `research/moe_lit_review.md` | Literature notes on MoE and hierarchical ICD coding |

## Data

No data is stored in this repo. The scripts download it from the original sources:

- [MIMIC-IV Clinical Database Demo v2.2](https://physionet.org/content/mimic-iv-demo/2.2/) (PhysioNet, open access)
- [CodiEsp](https://zenodo.org/records/3837305) (CLEF eHealth 2020, CC BY 4.0)
- [MTSamples](https://huggingface.co/datasets/harishnair04/mtsamples) (Hugging Face mirror)

Full MIMIC-IV data must never be committed here: its data use agreement forbids sharing it.

## Running it

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
mkdir -p data/mimic-demo && cd data
curl -L -o codiesp.zip "https://zenodo.org/api/records/3837305/files/codiesp.zip/content" && unzip -q codiesp.zip -d codiesp
curl -L -o mtsamples.csv https://huggingface.co/datasets/harishnair04/mtsamples/resolve/main/mtsamples.csv
for t in admissions diagnoses_icd services d_icd_diagnoses patients; do curl -L -o mimic-demo/$t.csv.gz https://physionet.org/files/mimic-iv-demo/2.2/hosp/$t.csv.gz; done
cd ../pilot
../.venv/bin/python mimic_demo.py && ../.venv/bin/python prepare.py && ../.venv/bin/python run_moe.py
```

The full comparison takes about 8 minutes on a laptop. For a fine-tuned encoder, run `pilot/kaggle_pilot.ipynb`
on a free Kaggle GPU (about 3 hours).
