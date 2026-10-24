# Object–Context Decoupled Pooling (OCDP) for Label-Free Mitigation of Shortcut Learning
## Project Analysis & System Blueprint

### 1. Project Overview & Problem Statement
Deep neural image classifiers frequently learn spurious correlations (shortcuts)—features that correlate with class labels in biased training distributions but fail to generalize to out-of-distribution or minority groups. On the canonical **Waterbirds** benchmark:
- 95% of waterbirds appear on water backgrounds;
- 95% of landbirds appear on land/forest backgrounds;
- Standard Empirical Risk Minimization (ERM) models using Global Average Pooling (GAP) conflate object features (bird morphology) and context features (water/land pixels) into a single unified representation vector;
- As a consequence, ERM achieves high overall accuracy (~95%) but suffers severely on minority groups (e.g., waterbirds on land, achieving worst-group accuracy of only ~60–75%).

#### Existing Approaches and Their Deficiencies
1. **Tier 1 (Training Group Labels Required)**: Group DRO and related methods require explicit spurious attribute annotations for every training sample ($g = 2y + a$). In real-world applications, collecting group annotations is prohibitively expensive or impossible.
2. **Tier 2 (Validation Group Labels Required)**: Deep Feature Reweighting (DFR) and JTT require group labels on the validation set for model selection, hyperparameter tuning, or last-layer retraining.
3. **Multi-stage & Post-hoc Tier 3 Approaches**: Methods like ExMap, Decompose-and-Compose (DaC), and MaskTune train an initial biased model, compute post-hoc explanations (Grad-CAM, LRP), cluster heatmaps or cut/paste image patches, and retrain a second model. These multi-stage pipelines are computationally heavy and depend on post-hoc explanation heuristics.
4. **Existing Spatial Attention**: Methods such as Deep Attention Reweighting (DAR) replace GAP with spatial attention, but require balanced held-out sets, lack a dedicated context sink, and are not trained end-to-end.

#### OCDP Objective
**Object–Context Decoupled Pooling (OCDP)** introduces an architectural inductive bias directly inside the pooling head of a convolutional neural network (ResNet-50) that:
1. Spatially decomposes local feature representations into an **Object Path** (retained at test time) and a **Context Sink** (active during training only);
2. Trains the Context Sink with Generalized Cross-Entropy (GCE, $q \approx 0.7$) to rapidly absorb background shortcut signals;
3. Dynamically reweights object-path training samples using the context sink's error ($w_i = \frac{\mathcal{L}_{\text{ctx}, i}}{\mathcal{L}_{\text{ctx}, i} + \mathcal{L}_{\text{obj}, i}}$), thereby assigning higher gradient weight to bias-conflicting images;
4. Enforces an **Area Prior** loss to prevent gate saturation and an optional **Counterfactual Consistency** loss;
5. Operates purely under **Tier 3 (no group labels in training or model selection)**, selecting models strictly via worst-class validation accuracy.

---

### 2. Workspace & Resource Findings
Inspection of the provided workspace (`d:\PROJECTS\internship`) revealed:
1. **Supplied Materials**:
   - `OCDP_Project_Handbook.docx`: Comprehensive 516-paragraph document detailing research background, objectives (O1–O6), research questions (RQ1–RQ4), architecture equations, training objectives, baseline tiers, and workflow stages.
   - `shortcut-starter-kit (1).zip`: Contains `AGENTS.md` (scientific integrity rules), `.agents/rules/evaluation-protocol.md`, `.agents/skills/stage-report/SKILL.md`, `docs/literature_matrix.md`, `docs/stage_prompts.md`, and `docs/OCDP_Research_Plan_v2.docx`.
2. **Environment & Hardware Audit**:
   - OS: Windows 11 (AMD64)
   - Python: 3.13.2
   - Core libraries: PyTorch 2.14.1+cpu, torchvision 0.29.1+cpu, PyArrow 26.0.0, NumPy 2.4.2, pandas 3.0.1, scikit-learn 1.8.0, SciPy 1.17.1, Matplotlib 3.10.8, PyYAML 6.0.3, pytest 9.0.2.
   - Accelerator: Intel(R) Iris(R) Xe Graphics (CPU execution environment for development, smoke tests, and pilot runs; full GPU execution scripts formatted for headless lab servers).
   - Storage: 129.8 GB free on `D:` drive.
3. **Dataset Status**:
   - No pre-extracted dataset was in the zip.
   - The canonical Stanford NLP URL (`nlp.stanford.edu/data/dro/waterbird_complete95_forest2water2.tar.gz`) is subject to network timeouts.
   - Exact official mirror verified: Hugging Face official dataset mirror `grodino/waterbirds` provides exact byte-level parity with the Group DRO splits (4,795 train, 1,199 val, 5,794 test) with verified class (`0=landbird, 1=waterbird`) and place (`0=land, 1=water`) metadata.

---

### 3. Architecture Specification

For a ResNet-50 backbone, the final convolutional feature map $F \in \mathbb{R}^{B \times C \times H \times W}$ (where $C = 2048, H = W = 7$ for $224 \times 224$ inputs) provides local feature vectors $f_u \in \mathbb{R}^C$ at each spatial grid location $u \in \{1, \dots, HW\}$.

#### 3.1 OCDP Head Components
1. **Evidence Head ($W_o, b_o$)**:
   - $1 \times 1$ convolution mapping $C \to K$ (where $K=2$ for Waterbirds):
     $$z_u = W_o f_u + b_o \in \mathbb{R}^K$$
   - Represents the local classification logits at grid position $u$.
2. **Gate Head ($g_\theta$)**:
   - Lightweight $1 \times 1$ convolutional MLP ($C \to 256 \to 1$) with intermediate ReLU:
     $$a_u = \sigma(W_{g2} \operatorname{ReLU}(W_{g1} f_u + b_{g1}) + b_{g2}) \in (0, 1)$$
   - Represents the probability/proportion of location $u$ belonging to the core object.
3. **Object Pooling (Inference & Prediction)**:
   - Soft-gated spatial average of evidence logits:
     $$\hat{y}_{\text{obj}} = \frac{\sum_u a_u z_u}{\sum_u a_u + \epsilon}$$
   - When all $a_u = 1$, this mathematically reduces to standard $\text{GAP} + \text{linear}$, ensuring stable initialization.
4. **Context Head & Sink ($W_c, b_c$)**:
   - Separate $1 \times 1$ convolution mapping $C \to K$:
     $$c_u = W_c f_u + b_c \in \mathbb{R}^K$$
   - Context pooling over the spatial complement of the object gate:
     $$\hat{y}_{\text{ctx}} = \frac{\sum_u (1 - a_u) c_u}{\sum_u (1 - a_u) + \epsilon}$$

#### 3.2 Loss Formulations
The total training loss is:
$$\mathcal{L} = \mathcal{L}_{\text{obj}} + \lambda_g \mathcal{L}_{\text{ctx}} + \lambda_a \mathcal{L}_{\text{area}} + \lambda_c \mathcal{L}_{\text{cons}}$$

1. **Object Loss with Dynamic Context-Error Reweighting**:
   $$\mathcal{L}_{\text{obj}} = \frac{\sum_i w_i \operatorname{CE}(\hat{y}_{\text{obj}, i}, y_i)}{\sum_i w_i}$$
   where the per-sample weight $w_i$ is determined by the relative error of the context sink:
   $$w_i = \operatorname{clip}\left(\frac{\operatorname{CE}(\hat{y}_{\text{ctx}, i}, y_i)}{\operatorname{CE}(\hat{y}_{\text{ctx}, i}, y_i) + \operatorname{CE}(\hat{y}_{\text{obj}, i}, y_i)}, w_{\min}, w_{\max}\right)$$
   Bias-conflicting samples (e.g. waterbirds on land) cause high context loss $\operatorname{CE}_{\text{ctx}}$, thus receiving elevated weights $w_i \gg 0.5$.
2. **Context Sink Loss (GCE)**:
   $$\mathcal{L}_{\text{ctx}} = \frac{1}{B} \sum_{i=1}^B \operatorname{GCE}_q(\hat{y}_{\text{ctx}, i}, y_i)$$
   $$\operatorname{GCE}_q(p, y) = \frac{1 - p_y^q}{q}, \quad q \in (0, 1], \text{ default } q = 0.7$$
   GCE implicitly amplifies gradients on easy, clean, shortcut-aligned examples, compelling the sink to latch onto background features.
3. **Area Regularization**:
   $$\mathcal{L}_{\text{area}} = \frac{1}{B} \sum_{i=1}^B (\bar{a}_i - \rho)^2, \quad \bar{a}_i = \frac{1}{HW} \sum_u a_{i, u}$$
   Target foreground ratio $\rho \approx 0.30$, preventing trivial collapse where $a_u \to 0$ or $a_u \to 1$.
4. **Counterfactual Consistency**:
   $$\mathcal{L}_{\text{cons}} = \frac{1}{B} \sum_{i=1}^B \operatorname{KL}\left(p_{\text{obj}}(x_i) \parallel p_{\text{obj}}(x_i \odot M(a_i))\right)$$
   where $M(a_i)$ is the upsampled soft/binary gate mask.

---

### 4. Baseline & Comparison Taxonomy
To rigorously answer Research Question 1 (RQ1) and ensure scientific integrity across tiers:

| Tier | Supervision | Methods | Model Selection Rule |
| :--- | :--- | :--- | :--- |
| **Tier 1 (T1)** | Group labels in train & val | Group DRO, SCER | Val Worst-Group Accuracy (WGA) |
| **Tier 2 (T2)** | Group labels in val only | DFR, JTT, Decompose-and-Compose (DaC), HSFM, POTER | Val Worst-Group Accuracy (WGA) |
| **Tier 3 (T3)** | **No group labels (class labels only)** | **OCDP (Ours)**, ERM, LfF, EVaLS, ExMap, ShortcutProbe | **Val Worst-Class Accuracy (WCA)** |

#### Architectural Controls (Matched Tier 3 Recipe)
To prove that performance gains originate from the decoupled architecture rather than incidental hyperparameters:
1. **GAP**: Standard global average pooling baseline.
2. **ABMIL**: Attention-based Multiple Instance Learning pooling (Ilse et al.).
3. **Top-K**: Top-$k$ spatial logit pooling (FocusMIL control).
4. **DAR-Style**: Deep Attention Reweighting spatial attention head without context sink.
5. **LfF-on-GAP**: Global auxiliary biased model with GCE on a standard GAP backbone to isolate reweighting effects.

---

### 5. Implementation Roadmap & Milestones
- **Phase 1: Environment & Project Foundation**
  - Dependency pinning (`requirements.txt`), git tracking, `.gitignore`.
  - Directory architecture (`src/data/`, `src/models/`, `src/losses/`, `src/baselines/`, `src/evaluation/`, `configs/`, `tests/`, `results/`).
- **Phase 2: Dataset Integration & Audit**
  - Implement `src/data/waterbirds.py` supporting both raw image directories and official parquet archives.
  - Verification tests (`tests/test_waterbirds.py`) for split counts, label/place mappings, and group ID calculation $g = 2y + \text{place}$.
  - Data integrity audit: hash duplicate checking, cross-split leakage verification, class/group distribution tabulation.
  - Evaluation-only isolation: strict boundary guard preventing training code from accessing group labels or segmentation masks.
- **Phase 3: Core Model & Head Implementations**
  - Implement ResNet-50 feature extraction backbone.
  - Implement all 5 pooling heads in `src/models/heads.py`: `GAPHead`, `ABMILHead`, `TopKHead`, `DARHead`, and `OCDPHead`.
  - Implement all losses in `src/losses/`: `CrossEntropyLoss`, `GeneralizedCrossEntropyLoss`, `AreaPriorLoss`, `ConsistencyLoss`, and dynamic weight calculator.
- **Phase 4: Smoke Testing & Unit Verification**
  - Comprehensive unit test suite in `tests/` covering:
    - Head parameter shapes, forward pass outputs, and gradient backpropagation.
    - Loss calculations, numerical stability ($\epsilon$ handling, GCE $q \to 0$ limits).
    - Weight clipping and EMA smoothing.
    - Evaluation metrics (WGA, MGA, average accuracy, confusion matrices).
- **Phase 5: Baseline & Pilot (P1) Execution**
  - ERM baseline training and validation logging.
  - Frozen feature extraction pipeline for rapid, reproducible head comparison across 3 seeds.
  - Benchmarking of GAP, ABMIL, Top-K, DAR, and OCDP under identical T3 evaluation protocols.
- **Phase 6: End-to-End OCDP Training & Baselines**
  - End-to-end training pipeline with warm-up epochs, dynamic sample reweighting, and area prior.
  - Built-in shortcut scoring ($s_i = p_{\text{ctx}}(y_i | x_i)$) and AUROC evaluation for minority group detection.
  - Faithful evidence map generation ($a_u \cdot z_u[y]$) and energy-in-mask localization evaluation.
- **Phase 7: Comprehensive Documentation & Reporting**
  - `README.md`, `REQUIREMENTS_CHECKLIST.md`, `TESTING.md`, stage reports, and experiment reproducibility logs.
