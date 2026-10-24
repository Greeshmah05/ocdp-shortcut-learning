# Verified literature matrix (as of 2026-10-10)

The coding agent may cite ONLY papers in this file. "Checked" says how far each entry was verified:
read = paper/page opened; title = title and venue confirmed from search results; memory = verify before citing.
Waterbirds worst-group accuracy (WGA, %) is as reported on the cited page.

## Supervision tiers
- T1: group labels in training. T2: group labels only on validation (tuning/selection). T3: no group labels.
- "Masks" = needs foreground/object segmentation masks.

## Methods

| Work | Venue | Core idea | Tier / extra supervision | Waterbirds WGA | Code | Checked |
|---|---|---|---|---|---|---|
| Group DRO (Sagawa et al.) | ICLR 2020 | Minimise worst-group loss with strong regularisation | T1 | 91.4 (reported in DaC) | https://github.com/kohpangwei/group_DRO | title |
| JTT (Liu et al.) | ICML 2021 | Upweight examples misclassified by a short ERM run | T2 | 86.7 | https://proceedings.mlr.press/v139/liu21f.html | title |
| DFR (Kirichenko, Izmailov, Wilson) | ICLR 2023 | Retrain last layer on group-balanced held-out data | T2 | 92.1-92.3 | https://openreview.net/pdf?id=Zb6c8A-Fghk | title |
| AFR (Qiu et al.) | ICML 2023 | Confidence-based reweighting for last-layer retraining | T2 | 88.7 ± 4.2 (in ShortcutProbe) | https://proceedings.mlr.press/v202/qiu23c/qiu23c.pdf | title |
| ShortcutProbe (Zheng, Ye, Zhang) | IJCAI 2025 | Probe latent shortcut directions, retrain for invariance; selects on worst-class acc | T3 | 90.8 ± 0.6 | https://www.ijcai.org/proceedings/2025/795 | read |
| ExMap (Chakraborty et al.) | CVPR 2024 | Cluster LRP heatmaps of ERM model into pseudo-groups, plug into DFR/JTT | T3 | 92.5 (DFR+ExMap) | https://arxiv.org/abs/2403.13870 | read |
| Decompose-and-Compose, DaC (Noohdani et al.) | CVPR 2024 | xGradCAM adaptive masks split causal/non-causal parts; compose cross-class images; retrain last layer | T2 (val for tuning) | 92.3 | https://arxiv.org/abs/2402.18919 | read |
| MaskTune (Asgari et al.) | NeurIPS 2022 | Mask xGradCAM-salient regions, fine-tune one epoch | T3 for training | check paper | https://arxiv.org/abs/2210.00055 | title |
| Discover and Cure (Wu et al.) | ICML 2023 | Concept-based discovery and intervention on spurious concepts | concept bank | check paper | https://proceedings.mlr.press/v202/wu23w.html | title |
| ISNet / LRP background relevance minimisation (Bassi, Dertkigil, Cavalli) | Nature Communications 15:291, 2024 | Penalise LRP relevance on background during training | Masks | not tested | https://www.nature.com/articles/s41467-023-44371-z | read |
| OCIE: Augmenting model interpretability via Deconfounded Explanation-Guided Learning | Knowledge-Based Systems, 2024 | Self-supervised ViT object discovery + class prototypes; align explanations to objects | self-supervised masks | not tested (ImageNet-100, CUB, Cars, HAM) | https://github.com/DLAIResearch/OCIE | read (README) |
| Deep Attention Reweighting (Chew, Wang) | arXiv 2605.20732, May 2026 | Replace GAP with post-hoc spatial attention, retrain with head (DFR extension) | as DFR | not in abstract | https://arxiv.org/abs/2605.20732 | read (abstract) |
| POTER (Jo, Kim, Yun, Chae) | arXiv 2610.01028, Oct 2026 | Optimal-transport-potential sample reweighting vs. validation reference | T2 | 90.9 ± 0.3 | https://arxiv.org/abs/2610.01028 | read |
| Bias Leaves a Gradient Trail | arXiv 2605.28780, May 2026 | Gradient probes on concept decompositions find spurious concepts without labels | T3 | not checked | https://arxiv.org/abs/2605.28780 | title |

## Detection / analysis / benchmarks

| Work | Venue | Relevance | Link | Checked |
|---|---|---|---|---|
| Explainer Divergence Scores | arXiv 2211.07650 | Can post-hoc explanations detect unknown spurious correlations (overlaps Candidate A) | https://arxiv.org/abs/2211.07650 | title |
| An XAI-based Analysis of Shortcut Learning in Neural Networks | arXiv 2504.15664 | Likely overlaps Candidate A; NOT YET READ | https://arxiv.org/abs/2504.15664 | title |
| Reassessing the Validity of Spurious Correlations Benchmarks (Bell, Bouchacourt, Sagun) | arXiv 2409.04188 | Waterbirds valid; CelebA, ImageNetBG fail validity checks | https://arxiv.org/abs/2409.04188 | read |
| Spurious Correlations in Machine Learning: A Survey (Ye et al.) | arXiv 2402.12715 | Related-work taxonomy | https://arxiv.org/abs/2402.12715 | title |
| Background Challenge / ImageNet-9 (Xiao et al.) | ICLR 2021 | Only-FG / only-BG / mixed-BG tests | - | memory |
| Right for the Right Reasons (Ross et al.) | IJCAI 2017 | Explanation-penalty loss | - | memory |
| GALS (Petryk et al.) | CVPR 2022 | Language-guided attention, Waterbirds | - | memory |
| Sanity Checks for Saliency Maps (Adebayo et al.) | NeurIPS 2018 | Saliency maps can be weight-independent | - | memory |
| SubpopBench / "Change is Hard" (Yang et al.) | ICML 2023 | Shared baseline codebase incl. Waterbirds | - | memory |

## Known numbers to keep in mind
- Published ERM WGA on Waterbirds varies: 70.8 (DaC), 76.8 (ExMap), 80.3 ± 3.1 (ShortcutProbe).
- Label-free (T3) state of the art is ~90.8-92.5 WGA; headroom is small.

## Added in v2: architecture-level and recent work (see plan v2, Section 3, for full tables)

| Work | Venue | Core idea | Supervision | WB WGA | Link | Checked |
|---|---|---|---|---|---|---|
| Deep Attention Reweighting (Chew, Wang) | arXiv 2605.20732 | Post-hoc spatial attention replaces GAP; 1,000-image balanced target set; main runs ResNet-18 from scratch | balanced set; minority val early stopping | 74.6 (DFR 73.3) | https://arxiv.org/abs/2605.20732 | read |
| Rethinking GAP as MIL (Karjauv) | arXiv 2606.14555 | GAP + linear = mean of per-location logits; names learned background-instance component as future work | - | none | https://arxiv.org/abs/2606.14555 | read |
| Two-stage ViT hard masking (Aniraj et al.) | ICPR 2026 | Stage 1 finds parts; stage 2 sees masked input | not stated | not reported | https://github.com/ananthu-aniraj/ifam | read (repo) |
| LfF (Nam et al.) | NeurIPS 2020 | GCE-trained biased model; up-weight hard examples | T3 | check | https://arxiv.org/abs/2007.02561 | title |
| ReBias (Bahng et al.) | ICML 2020 | Separate small-receptive-field biased model | T3 | - | https://proceedings.mlr.press/v119/bahng20a/bahng20a.pdf | title |
| DisEnt (Lee et al.) | NeurIPS 2021 | Two encoders, intrinsic vs bias features | T3 | - | https://arxiv.org/abs/2107.01372 | title |
| EVaLS (Ghaznavi et al.) | arXiv 2410.05345 | Loss-balanced last-layer retraining | T3 | 88.4 ± 3.1 | https://arxiv.org/abs/2410.05345 | read |
| AGS (ICLR 2026) | ICLR 2026 | Alignment-gated suppression; channel/head gating as future work | T3 | 80.9 | https://en.papernotes.org/ICLR2026/others/regulating_internal_alignment_flows_for_robust_learning_under_spurious_correlati/ | read (notes) |
| HSFM (Yazdan Parast et al.) | arXiv 2603.29313 | Meta-learned support embeddings vs hard val set | T2 | 93.1 ± 0.1 | https://arxiv.org/abs/2603.29313 | read |
| SCER (Park et al.) | arXiv 2511.04401 | Spurious-direction embedding regulariser | T1 | 91.2 ± 0.2 | https://arxiv.org/abs/2511.04401 | read |
| SupER (Liu et al.) | CPAL 2026 | beta-VAE relevant/irrelevant latent split, CLIP guidance | superclass + CLIP | 84.4 ± 2.3 | https://arxiv.org/abs/2508.08570 | read |
| FocusMIL (Liu et al.) | arXiv 2408.09449 | Max-pooling MIL more robust than attention MIL | - | - | https://arxiv.org/abs/2408.09449 | read |
| Memorization neurons (You et al.) | arXiv 2501.00961 | Critical neurons drive group gap | T3 | 80.9 (R50) | https://arxiv.org/abs/2501.00961 | read (review) |
| ASM (Kuhn et al.) | arXiv 2501.00942 | ViT token clustering + patch ablation | T3 | not on WB | https://arxiv.org/abs/2501.00942 | read |
| Rao et al. | ICCV 2023 | Explanation guidance with boxes; B-cos | boxes | WB-100: 43.4 -> 56.1 | https://arxiv.org/abs/2303.11932 | read |
| UrbanCars / Whac-A-Mole (Li et al.) | CVPR 2023 | Multi-shortcut benchmark | - | - | https://arxiv.org/abs/2212.04825 | title |
| Box-Guided Evidence Routing, BGER (Liang et al.) | arXiv 2610.05891, Oct 2026 | Gate on final feature map trained with boxes/masks to improve post-hoc CAMs; CUB, Stanford Dogs, chest X-ray | boxes or masks | not tested | https://arxiv.org/abs/2610.05891 | read (abstract) |
