# Literature Review: MoE and Hierarchical ICD Coding (MIMIC-III/IV)

Compiled 2026-10-05. Every paper below was opened at the listed URL. Numbers marked **unverified** come only from search snippets or abstracts, not from a results table I read.

## TL;DR

- **No paper found that compares experts partitioned by ICD chapter against experts partitioned by specialty or hospital service.** The closest MoE paper for ICD coding (GoM-ICD, 2025) uses learned experts, and its abstract does not tie them to chapters or specialties. This looks like a real research gap.
- **Hierarchy gives small, consistent gains, mostly on rare codes.** On MIMIC-III-full, JointLAAT raised macro-F1 from 9.9 to 10.7 with micro-F1 unchanged at 57.5. The Two-Stage Decoder gained +0.9 micro-F1 over JointLAAT, and HyperCore's hierarchy component was worth about +1.2 micro-F1.
- **Conditioning child codes on parent predictions already exists, but nobody has done it with MoE.** JointLAAT concatenates an embedding of the parent prediction into each child classifier. The Two-Stage Decoder lets child labels attend to parent labels. I found no "stacked MoE" or FiLM-style version.
- **The "+15 points from multiple models" claim is not supported.** The best ensemble result I found (Yang et al., AAAI 2023) adds +0.9 micro-F1 and +4.2 macro-F1 on MIMIC-III-full.
- **No ICD-coding paper found that uses the MIMIC `services.curr_service` field or note specialty as a feature or router.** The only department-split study (Deep-ADCA) uses private hospital data and per-department evaluation, not routing.

## Q1: MoE and multi-expert models for ICD coding

- **GoM-ICD** (Wu et al., *Big Data Mining and Analytics*, 2025) is the only ICD-coding MoE paper I found. It stacks text-level and label-level MoE layers on a PLM+BiLSTM encoder. Reported micro-F1 is 61.7 on MIMIC-III full, 62.0 on MIMIC-III clean and 61.3 on MIMIC-IV ICD-10. The landing page does not say how the experts are partitioned or give an MoE ablation (**unverified**; check the PDF or code at github.com/CSUBioGroup/GoM-ICD).
- **Clinical MoE outside ICD coding** routes by **modality**, not by specialty:
  - FuseMoE (NeurIPS 2024)
  - MoE-Health (2025)
  - LongMoE (2026; it has dedicated experts for notes, labs, ICD codes and demographics on MIMIC-IV)
  - Med-MoE (EMNLP Findings 2024; domain-specific experts in a medical vision-language model)
- **Department-level modelling:** Deep-ADCA (Masud et al., *J. Pers. Med.* 2022) evaluates a CNN across 5 outpatient departments using private Taiwanese hospital data, not MIMIC. Precision was 0.50–0.69 and recall 0.78–0.91, with cardiology performing best. This is per-department evaluation, not routing.
- **Chapter vs specialty comparison: none found.** Searches for chapter-gated experts, specialty-gated experts and service-routed ICD models returned nothing directly relevant.

## Q2: Hierarchical (coarse-to-fine) ICD coding

Watch the naming: several "hierarchical" models are hierarchical over the **document**, not over the **code tree**.

| Model | Hierarchy over | Reported gain from hierarchy |
|---|---|---|
| JointLAAT (IJCAI 2020) | Code tree: 3-character parent, then full code | MIMIC-III-full: macro-F1 9.9→10.7 (p<0.01), micro-F1 57.5→57.5. MIMIC-II: macro-F1 5.9→6.8, micro-F1 48.6→49.1. MIMIC-III-50: no gain (micro-F1 71.5 vs 71.6), because only 8 of 40 parents have more than one child. |
| Two-Stage Decoder (ACL Findings 2023) | Code tree: parent, then child | Compared with JointLAAT on MIMIC-III-full: micro-F1 57.5→58.4, macro-F1 +0.3, P@8 73.5→74.4. On MIMIC-III-50: micro-F1 71.83. Gains concentrate on rare codes. |
| HyperCore (ACL 2020) | Code tree (hyperbolic embedding) plus a co-occurrence graph | Removing the hyperbolic part on MIMIC-III-full: micro-F1 55.1→53.9 and macro-F1 9.0→8.1. |
| HiCu (MLHC 2022) | Code tree, used as a training curriculum | Improves recurrent, convolutional and transformer models. Exact gains **unverified** (not in the abstract). |
| XR-LAT (JBI 2023) | Code tree (recursive model chain) | Competitive with PLM-ICD, with macro-AUC +2.1 on MIMIC-III and +5.1 on MIMIC-II (abstract). The best micro-F1, 60.8, came from a tuned PLM-ICD, not from XR-LAT. |
| Graph-Constrained Policy (arXiv 2026) | ICD-10-CM tree, decoded one level at a time with an LM | MIMIC-IV full: micro-F1 52.7, which is +4.4 micro-F1 and +15.7 macro-F1 over flat CAML/LAAT/PLM-ICD (abstract summary; preprint). |
| HiLAT (JBI 2022) | **Document**: token level, then chunk level | MIMIC-III-50: micro-F1 73.5 and macro-F1 69.0, about +1.8 micro over ISD. Not code hierarchy. |
| HLAN (JBI 2021) | **Document**: word level, then sentence level | MIMIC-III-50: micro-F1 64.1 (from the HiLAT comparison table). Not code hierarchy. |

**Generative and encoder-decoder coding:** GPsoap (Yang et al., AAAI 2023) generates code descriptions autoregressively. It reaches macro-F1 30.2 on MIMIC-III-few, against 18.7 for the previous best model.

## Q3: Finer prediction conditioned on coarser prediction

- **JointLAAT:** the vector of parent probabilities is projected to a vector s_D, which is concatenated with every child's label-specific vector before that child's feed-forward classifier. Training minimises the sum of the parent and child cross-entropy losses. This is the simplest baseline for this idea.
- **Two-Stage Decoder:** child labels attend over both the parent-label representations and the document. The authors frame this explicitly as better than JointLAAT's concatenation. It also improves parent prediction itself (parent micro-F1 68.1→69.0).
- **Graph-Constrained Policy:** each depth decision is conditioned on the full path chosen so far.
- **Not found:** gating, MoE or FiLM conditioning in which the experts get more specific as code depth increases. Condition-CNN (fashion images, *Expert Systems with Applications* 2021) feeds coarse predictions as priors to finer levels, so it is a cross-domain precedent only.

## Q4: "AX-something"

I could not identify a single definite match. Plausible candidates, in my rough order of likelihood:

1. **AttentionXML** (You et al., NeurIPS 2019): a label-tree extreme-classification model. It is often written "AXML" and is the base design that ICDXML builds on.
2. **XR-LAT / XR-Transformer** (Liu et al., JBI 2023): hierarchical code-tree models evaluated on MIMIC.
3. **ICDXML** (*Scientific Reports* 2024): probabilistic label trees for ICD coding on MIMIC-III.

Ask the teammate for the paper link.

## Q5: "Multiple models add at least 15 points"

There is no support for this in ICD-coding ensemble work I could verify:

- **GPsoap ensemble reranker** (MSMN + GPsoap, AAAI 2023) on MIMIC-III-full: micro-F1 58.2→59.1 (+0.9) and macro-F1 10.4→14.6 (+4.2). This is the strongest verified ensemble gain.
- **Heliyon 2024 ensemble** (text models plus random forest or boosting on structured data): the abstract says it beats Longformer and knowledge-graph baselines, with no exact delta given. A search snippet quoted micro-F1 0.402→0.475 (+7.3) for an ensemble, which is **unverified** (full text was blocked).
- **Sen et al. 2021** report +18% micro-F1. That comes from reformulating the task (sentence tagger plus classifier) on private phrase-annotated data, is not an ensemble, and is probably relative rather than absolute (**unverified**).
- A search snippet claimed GatorTron gains of "+11%/+15.4%". These are **unverified** and appear to be relative improvements from a single model, not an ensemble.

**Caution:** Edin et al. (SIGIR 2023) show that many published MIMIC gaps shrink under correct splits and metric computation. In particular, macro-F1 was computed incorrectly in prior work, so large claimed gains deserve scepticism.

## Q6: The services table (`curr_service`) or note specialty as a feature or router

None found for ICD coding. The standard benchmarks (CAML/Mullenbach splits on MIMIC-III; Edin et al. and Nguyen et al. "Mimic-IV-ICD" on MIMIC-IV) use the discharge summary text only. Wang et al. (NAACL 2024) is the nearest precedent for structured side-information: they retrieve candidate codes using DRG codes, CPT codes and prescriptions. Removing that auxiliary knowledge dropped micro-F1 from 60.3 to 57.9 on MIMIC-III-full. Service would be an analogous structured signal.

## Paper table

| Title (short) | Authors | Year | Venue | URL | Key result | Relevance |
|---|---|---|---|---|---|---|
| LAAT / JointLAAT | Vu, Nguyen, Nguyen | 2020 | IJCAI | https://www.ijcai.org/proceedings/2020/0461.pdf | Full: micro-F1 57.5, macro-F1 10.7 (Joint) | Q2, Q3 baseline |
| Two-Stage Decoder | T.-T. Nguyen, Schlegel, Kashyap, Winkler | 2023 | ACL Findings | https://arxiv.org/abs/2306.00005 | Full: micro-F1 58.4; 50: 71.83 | Q2, Q3 |
| HyperCore | Cao, Chen, Liu, Zhao, Liu, Chong | 2020 | ACL | https://aclanthology.org/2020.acl-main.282/ | Full: micro-F1 55.1; hierarchy worth +1.2 | Q2 |
| HiCu | Ren, Zeng, Wu, Zhu, Krishnan | 2022 | MLHC (PMLR 182) | https://arxiv.org/abs/2208.02301 | Hierarchy curriculum helps all architectures | Q2 |
| XR-LAT | Liu, Perez-Concha, Nguyen, Bennett, Jorm | 2023 | J. Biomed. Inform. 144 | https://arxiv.org/abs/2212.05857 | PLM-ICD tuned: micro-F1 60.8 | Q2, Q4 |
| HiLAT | Liu et al. | 2022 | J. Biomed. Inform. 133 | https://arxiv.org/abs/2204.10716 | 50: micro-F1 73.5 | Q2 (document hierarchy) |
| HLAN | Dong, Suárez-Paniagua, Whiteley, Wu | 2021 | J. Biomed. Inform. | https://arxiv.org/abs/2010.15728 | 50: micro-F1 64.1 | Q2 (document hierarchy) |
| Graph-Constrained Policy | Singh, Torres, Shakeel, Bukhari | 2026 | arXiv | https://arxiv.org/abs/2607.11954 | MIMIC-IV full: micro-F1 52.7 | Q2, Q3 |
| GoM-ICD | Wu, Qiu, Zeng, Chen, Li, Zhu | 2025 | Big Data Min. Anal. 8(6) | https://www.sciopen.com/article/10.26599/BDMA.2025.9020019 | micro-F1 61.7 / 62.0 / 61.3 | Q1 (only ICD MoE) |
| GPsoap | Yang, Kwon, Yao, Yu | 2023 | AAAI | https://ojs.aaai.org/index.php/AAAI/article/view/25668 | Ensemble: micro-F1 +0.9, macro-F1 +4.2 | Q2 generative, Q5 |
| Ensemble neural models | Merchant, Shenoy, Lanka, Kamath | 2024 | Heliyon | https://doi.org/10.1016/j.heliyon.2024.e36569 | Text plus structured ensemble; delta unverified | Q5 |
| Multi-stage Retrieve and Re-rank | X. Wang, Mercer, Rudzicz | 2024 | NAACL | https://aclanthology.org/2024.naacl-long.273/ | micro-F1 60.3; auxiliary data worth +2.4 | Q6 |
| Deep-ADCA | Masud et al. | 2022 | J. Pers. Med. | https://doi.org/10.3390/jpm12050707 | Per-department precision 0.50–0.69 | Q1, Q6 (non-MIMIC) |
| Replicability study | Edin et al. | 2023 | SIGIR | https://arxiv.org/abs/2304.10909 | Fixes splits and macro-F1; first MIMIC-IV baselines | Evaluation protocol |
| From XML to Multi-class | Sen, Ye, Aslam, Tahmasebi | 2021 | arXiv | https://arxiv.org/abs/2102.09136 | +18% micro-F1 (private data) | Q5 caveat |
| Med-MoE | Jiang et al. | 2024 | EMNLP Findings | https://aclanthology.org/2024.findings-emnlp.221/ | Domain-specific medical experts (vision-language) | Q1 (non-ICD) |

## Gaps and where our project could contribute

1. **Chapter-routed vs service-routed experts.** No published head-to-head comparison exists. A controlled comparison on the same backbone and the same Edin et al. MIMIC-IV split would be novel. It should include a learned-router baseline, in the style of GoM-ICD.
2. **`curr_service` as a router or prior.** This is unused in ICD-coding work. The router choices to test are hard routing (service → expert), soft routing (service embedding → gate) and service plus text.
3. **Depth-wise MoE / FiLM conditioning.** Chapter-level predictions would gate or FiLM-modulate category-level experts, which in turn condition the full-code heads. JointLAAT (concatenation) and the Two-Stage Decoder (attention) are the right baselines.
4. **Honest evaluation.** Report micro-F1, macro-F1 and P@k plus rare-code buckets, using the Edin et al. protocol. Expect gains of about 1–4 points, not 15. The hierarchy literature shows gains mostly in macro-F1 and on rare codes.
