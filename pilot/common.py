"""Shared helpers: ICD-9/10 chapter mapping, specialty grouping, paths."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
OUT = ROOT / "pilot" / "out"
OUT.mkdir(parents=True, exist_ok=True)
CODIESP = DATA / "codiesp" / "final_dataset_v4_to_publish"

# ICD-10-CM chapters: (first code, last code, short name). Compared on the first 3 chars.
ICD10_CHAPTERS = [
    ("A00", "B99", "Infectious"), ("C00", "D49", "Neoplasms"), ("D50", "D89", "Blood/Immune"),
    ("E00", "E89", "Endocrine"), ("F01", "F99", "Mental"), ("G00", "G99", "Nervous"),
    ("H00", "H59", "Eye"), ("H60", "H95", "Ear"), ("I00", "I99", "Circulatory"),
    ("J00", "J99", "Respiratory"), ("K00", "K95", "Digestive"), ("L00", "L99", "Skin"),
    ("M00", "M99", "Musculoskeletal"), ("N00", "N99", "Genitourinary"), ("O00", "O9A", "Pregnancy"),
    ("P00", "P96", "Perinatal"), ("Q00", "Q99", "Congenital"), ("R00", "R99", "Symptoms"),
    ("S00", "T88", "Injury"), ("U00", "U85", "Special"), ("V00", "Y99", "External"),
    ("Z00", "Z99", "Health status"),
]
CHAPTER_NAMES = [c[2] for c in ICD10_CHAPTERS]

# ICD-9-CM numeric ranges mapped onto the ICD-10 chapter names above, so both versions share one axis.
ICD9_RANGES = [
    (1, 139, "Infectious"), (140, 239, "Neoplasms"), (240, 279, "Endocrine"), (280, 289, "Blood/Immune"),
    (290, 319, "Mental"), (320, 359, "Nervous"), (360, 379, "Eye"), (380, 389, "Ear"),
    (390, 459, "Circulatory"), (460, 519, "Respiratory"), (520, 579, "Digestive"),
    (580, 629, "Genitourinary"), (630, 679, "Pregnancy"), (680, 709, "Skin"),
    (710, 739, "Musculoskeletal"), (740, 759, "Congenital"), (760, 779, "Perinatal"),
    (780, 799, "Symptoms"), (800, 999, "Injury"),
]


def icd10_chapter(code: str) -> str | None:
    c = code.replace(".", "").upper()[:3]
    for lo, hi, name in ICD10_CHAPTERS:
        if lo <= c <= hi:
            return name
    return None


def icd9_chapter(code: str) -> str | None:
    c = code.replace(".", "").upper()
    if c.startswith("V"):
        return "Health status"
    if c.startswith("E"):
        return "External"
    try:
        n = int(c[:3])
    except ValueError:
        return None
    for lo, hi, name in ICD9_RANGES:
        if lo <= n <= hi:
            return name
    return None


def icd_chapter(code: str, version: int = 10) -> str | None:
    return icd10_chapter(code) if version == 10 else icd9_chapter(code)


# MTSamples specialty -> MIMIC-style service group. Labels that are note types
# (Consult, SOAP, Discharge Summary...) or too small are dropped (absent from this map).
MTSAMPLES_TO_SERVICE = {
    "General Medicine": "MED",
    "Cardiovascular / Pulmonary": "CMED",
    "Surgery": "SURG",
    "Orthopedic": "ORTHO",
    "Neurology": "NMED",
    "Neurosurgery": "NSURG",
    "Gastroenterology": "GI",
    "Obstetrics / Gynecology": "GYN",
    "Urology": "GU",
    "Nephrology": "GU",
    "Hematology - Oncology": "OMED",
    "Psychiatry / Psychology": "PSYCH",
}
SERVICES = sorted(set(MTSAMPLES_TO_SERVICE.values()))


if __name__ == "__main__":
    assert icd10_chapter("s22.49xa") == "Injury"
    assert icd10_chapter("D50.9") == "Blood/Immune" and icd10_chapter("d49") == "Neoplasms"
    assert icd10_chapter("O9A.1") == "Pregnancy" and icd10_chapter("t88.7") == "Injury"
    assert icd9_chapter("4019") == "Circulatory" and icd9_chapter("V5861") == "Health status"
    assert icd9_chapter("E8490") == "External" and icd9_chapter("0389") == "Infectious"
    print("ok")
