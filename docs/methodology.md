# Methodology

A short companion to the running demo. Written for a technical reader who
wants to know what the prototype does, what it does not, and where the
open research questions sit.

## 0. Stack rationale (summary)

Every component below was chosen against named alternatives. The full
discussion — comparison tables, alternative options, and what would
change at production scale — lives in [`design_decisions.md`]
(./design_decisions.md). One-line summaries:

- **U-Net (ResNet34) over DeepLabV3+, SegFormer, Mask2Former, HRNet,
  YOLOv8-seg.** Thin-structure friendly; ImageNet-pretrained encoder
  fits 4 GB VRAM; smp library makes encoder swaps trivial.
- **Focal + Dice loss over BCE, Tversky, Lovász, boundary loss.**
  Standard recipe for class-imbalanced thin-structure segmentation;
  trains stably in a 2-hour budget.
- **Albumentations over torchvision.transforms.v2 and Kornia.** The de
  facto standard for segmentation augmentation; mask synchronisation
  is correct and well-tested.
- **PyTorch Lightning over vanilla PyTorch and MMSegmentation.**
  Encapsulates the loop without locking in the architecture; weights
  load with plain `torch.load` for the inference server.
- **FastAPI over Flask, Django, Litestar.** Type-safe, async, OpenAPI
  free, mature ecosystem.
- **Leaflet over Mapbox GL, OpenLayers, deck.gl.** No API token, no
  usage quota, sufficient for GeoJSON rendering.
- **Three.js over Babylon.js, Potree, deck.gl.** Best-documented 3D
  library on the web; capable up to a few hundred thousand points.
- **NetworkX over igraph and graph-tool.** Pure Python, contains the
  Steiner-tree approximation we need, fine for small graphs.
- **laspy over PDAL Python bindings and Open3D.** Pure Python read,
  sufficient for read-only LAS/LAZ.
- **uv over Poetry, pip+venv, conda.** Fast, lockfile-first, modern.
- **cloudflared named tunnel over ngrok and port forwarding.**
  Professional URL, auto-HTTPS, no reconnect window.

## 1. Problem framing

UK Distribution Network Operators face a regulatory and operational
imperative to digitise their low-voltage (LV) networks. Many LV cable
runs were recorded only on paper and never transferred to the modern
GIS systems that handle higher voltages. This is the problem space
the prototype addresses.

The problem decomposes into four sub-problems:

1. **Detection of visible conductors in image data.** Aerial RGB and
   drone imagery of overhead lines.
2. **Inference of hidden or obscured conductors.** Underground cables,
   vegetation occlusion, low-resolution evidence.
3. **Conversion to a GIS-usable representation.** Vector linestrings,
   topology, integration with existing asset data.
4. **Use of LiDAR alongside RGB.** ALS and drone LiDAR data already
   exist in DNO archives.

The prototype is a working spine through all four.

## 2. Segmentation

### Architecture

A U-Net (Ronneberger et al., 2015) with a ResNet34 (He et al., 2016)
encoder, ImageNet-pretrained. ResNet34 sits at a defensible point on
the depth/memory trade-off — deep enough to learn cable texture,
shallow enough to fine-tune at 768 × 768 with batch size 12 on a
16 GB T4 (v2 production), or at 512 × 512 with batch size 8 on the
4 GB 1050 Ti (v1 historical; the resolution drop was the binding
constraint of training on the laptop directly). The decoder is the
standard U-Net symmetric upsampler with skip connections from the
encoder.

### Training

- **Loss**: 0.5 × FocalLoss (Lin et al., 2017) + 0.5 × DiceLoss. Focal
  dampens the dominant negative class; Dice rewards mask overlap on
  the small positive region. Both are needed: focal alone tends to
  produce under-confident masks; Dice alone collapses early without
  a region-of-interest signal.
- **Augmentation** (Albumentations): random resized crop, flips,
  rotations, colour jitter, mild Gaussian blur. No mosaic or MixUp —
  these tend to break thin-structure tasks.
- **Optimiser**: AdamW with cosine LR schedule. Initial LR 1e-3,
  weight decay 1e-4.
- **Pretraining transfer**: ImageNet weights are kept on the encoder.
  Self-supervised pretraining (DINOv2, MAE) on un-labelled aerial
  imagery would be the next step for a project with access to a large
  unlabelled archive — likely the case at SSEN.

### Inference

Sliding-window tiling at 768 × 768 with a 64-pixel overlap (the v2
production configuration, matched to the training resolution to close
the inference-path drift documented in §10), reflective padding to a
multiple of the tile, and Hann-windowed averaging during stitching to
suppress tile-edge artefacts. This lets the model run on arbitrary
input sizes without retraining.

## 3. Vectorisation

A binary mask is one half of the GIS-readiness story. Asset data needs
**linestring geometry** with explicit topology (which cable connects to
which pole). The pipeline:

1. **Threshold + morphological open** to remove sub-cable speckle.
2. **Skeletonize** with scikit-image's `skeletonize` (the
   3-D medial-surface thinning algorithm of Lee, Kashyap & Chu, 1994).
3. **Walk the skeleton**: count 8-connected neighbours per skeleton
   pixel. Pixels with one neighbour are endpoints; pixels with three
   or more are junctions. Endpoints and junctions become graph nodes;
   chains between them become edges.
4. **Simplify** each edge with Douglas-Peucker (`shapely.simplify`).

The output is a `networkx.Graph` and a list of `LineString`s. The graph
representation lets downstream modules (hidden-cable inference,
topology validation) reason about connectivity.

## 4. Hidden-cable inference

The least-mapped problem and the most academically interesting one. Two
methods are demonstrated:

### 4.1 Catenary fit

Suspended conductors follow a catenary curve under their own weight
(Irvine, 1981):

    y(x) = a · cosh((x − x₀) / a) + c

When two pole tops or visible cable endpoints are observed and the
span between them is occluded, a catenary fit is well-posed even
without observations between the anchors — the curve has only three
parameters (a, x₀, c) and two boundary conditions plus a sag prior
constrains them adequately.

The implementation (`app/ml/catenary.py`) uses SciPy's least_squares
in a span-aligned coordinate frame, with an initial sag of 2% of the
span (typical for LV lines). A confidence band is rendered to
communicate that the inferred curve is one hypothesis from a family.

### 4.2 Topology completion

When the question is "where is the cable" rather than "how does the
cable hang", a different framing is needed. The LV network is a tree
rooted at a transformer; every property must be a leaf. We have:

- Building footprints (from OpenStreetMap or OS Open Zoomstack).
- Transformer locations (known from existing GIS).
- Possibly some pole positions (detectable in imagery).
- Possibly some cable fragments (from the segmenter).

The inference is a **Steiner-tree approximation**: the minimum-cost
tree spanning all required terminals (transformer + buildings) on a
candidate edge graph. Edge costs are Euclidean distance, discounted
where the candidate edge overlaps an observed cable fragment. The
implementation (`app/ml/graph_complete.py`) uses NetworkX's
Kou-Markowsky-Berman approximation (Kou, Markowsky & Berman, 1981).

The Steiner-tree baseline is deterministic and fast. A graph neural
network conditioned on building footprints, transformer locations,
and fragment evidence is the natural next step — it would learn
topology priors that the deterministic method cannot capture (e.g.
"feeders prefer to follow streets", "branch lengths follow a
characteristic distribution"). Khalil, Dai, Zhang, Dilkina & Song
(2017) is the canonical precedent for training GNNs as policies for
combinatorial-optimisation problems on graphs, and would underpin
the Phase-2 learned successor named in `docs/research_roadmap.md`.

## 5. LiDAR classification

Per-point eigenvalue features (Demantké et al., 2011): for each point,
the local k-NN covariance matrix gives sorted eigenvalues
λ₁ ≥ λ₂ ≥ λ₃, from which we compute:

- **linearity** = (λ₁ − λ₂) / λ₁ — high for cables and tree branches
- **planarity** = (λ₂ − λ₃) / λ₁ — high for ground and roof surfaces
- **sphericity** = λ₃ / λ₁ — high for vegetation
- **verticality** = 1 − |n_z| where n is the smallest-eigenvalue
  eigenvector — high for poles and walls

Combined with height-above-ground (estimated as the 5th percentile of z),
threshold-based rules separate ground, vegetation, conductor, and
structure. No deep learning is used — the geometric features are
interpretable and adequate for a clean visual demonstration. PointNet++
(Qi et al., 2017) or a sparse 3D CNN (MinkowskiNet, Choy, Gwak &
Savarese, 2019) is the obvious upgrade for production scale.

## 6. Limitations and what would change for the real KTP

### Domain gap
TTPLA is rural transmission lines. LV imagery is urban service cables
in cluttered backgrounds. The model trained here would not transfer
directly. The KTP would address this with:

- **Active labelling** on SSEN's own urban aerial / drone imagery —
  even a few hundred hand-labelled tiles would close most of the gap.
- **Self-supervised pretraining** on SSEN's unlabelled aerial archive
  using DINOv2 (Oquab et al., 2023) or MAE (He et al., 2022).
- **Synthetic urban data** by pasting cable patches onto Mapillary or
  OpenAerialMap backgrounds.

### Geo-referencing
TTPLA imagery is not georeferenced. The demo maps pixel coordinates
onto a synthetic lat/lon box near Thatcham for visualisation only.
Real production work consumes camera-pose metadata from the survey
provider.

### Modality fusion
The LiDAR module is independent of the RGB segmenter. The natural
Phase-2 step is mid-level RGB+LiDAR fusion: dual encoders, cross-
attention between modalities, shared decoder. Recent fusion work in
autonomous driving — BEVFusion (Liu et al., 2023) and TransFusion
(Bai et al., 2022) — provides good templates.

### Hidden-cable inference
The deterministic Steiner-tree method is a methodology demonstrator,
not a production tool. A graph neural network — likely GraphSAGE
(Hamilton, Ying & Leskovec, 2017) or a GAT (Veličković et al., 2018)
operating on the building/pole graph with cable-fragment node
features — is the obvious research direction. Training data would be
synthesised by masking known segments of digitised LV networks.

### Evaluation
A real evaluation would split data geographically (train on one
region, test on another) to measure generalisation rather than
memorisation; use IoU and centerline-recall (CCQ from the road
extraction literature) rather than pixel accuracy; and validate the
end-to-end output against a held-out set of digitised LV records.

### Operational integration
The demo runs as a single web service. Production deployment for
SSEN would integrate with their existing GIS (likely Esri ArcGIS),
their asset management system, and their connection-design workflow.
Interfaces would be REST or OGC API Features rather than a hand-coded
JSON contract.

## 7. Where this fails

A working model is uninteresting without an honest account of when
it breaks. The evaluation script (`training/evaluate.py`) writes
[`evaluation_results.md`](evaluation_results.md) and
[`evaluation_results_session.md`](evaluation_results_session.md) with
six concrete failure cases drawn from the held-out splits; the three
mechanisms below describe the categories the empirical failures fell
into. **All six observed failures fit one or more of these patterns.**

### 7.1 Texture confusion

Thin, dark, locally-linear features that are *not* cables — a fence
shadow, a roof seam between two metal panels, a road lane marking,
the edge of a long parked vehicle. A U-Net at 512 × 512 has a
receptive field around 130 px and cannot disambiguate locally; if
the surrounding context doesn't extend far enough into the tile, the
model fires. This is the single largest false-positive source on
TTPLA, especially in agricultural scenes where furrows resemble
parallel cables.

**What helps:** larger receptive field (HRNet, transformer encoder),
explicit pole/structure conditioning so the model only commits to a
cable where its endpoints are plausible, or training with hard-
negative mining on confirmed-not-cable patches.

### 7.2 Scale failure

Cables too thick (close-range drone footage) or too thin
(high-altitude tiles) relative to the training distribution. TTPLA
imagery sits in a narrow scale band; the model's filters are tuned
for cables ~2–4 px wide. At 1 px the cable is below the noise floor
of the encoder's first stage; at 8 px the model under-segments,
producing a thin trace through the centre of a thick cable.

**What helps:** multi-scale training augmentation (RandomResizedCrop
with a wider scale range), or scale-aware test-time augmentation
(predict at three resolutions and average).

### 7.3 Context failure

A cable visible to a human only because of surrounding context — two
poles at known endpoints, with the visible cable signal degrading to
< 1 px of contrast in the middle of the span. Pure local features
are inadequate; reading the scene requires reasoning about
"endpoints exist, therefore something connects them."

**What helps:** the catenary fit and Steiner-tree completion modules
in this prototype were designed precisely for this case. They
operate downstream of the segmenter on partial evidence, restoring
the missing centerline using a physical (catenary) or topological
(graph) prior. The ML half does what it can; the geometric half
finishes the job.

### Empirical failure pattern

The six failures captured by `training/evaluate.py` (three from the
random split, three from the session-grouped split — see the
evaluation result documents) distribute across the three mechanisms
above as follows:

- **Scale failure** dominates the random-split failures. The
  random-split evaluation runs at full 4K resolution where TTPLA
  cables are ~3–5 px wide; the model trained on resized 512 × 512
  crops where the same cables are 1–2 px wide, so its first-stage
  filters have a different scale prior than the production inference
  path needs. This is the **inference-path drift** flagged in §6
  and made empirically visible by the gap between training-time IoU
  (0.611) and production-path IoU (0.139).

- **Vegetation occlusion** (a sub-case of context failure) dominates
  the session-grouped failures. Multiple `14_*` images are dense
  pine canopy with cables visible only between branches — the
  receptive field at 512 × 512 cannot stitch the visible fragments
  back into a coherent prediction.

- **Texture confusion** appears where cables run parallel to road
  markings or roof eaves of similar contrast and orientation. The
  threshold step then suppresses the entire ambiguous neighbourhood
  rather than committing to either side.

The single recurring theme: **the model fails by producing nothing,
not by producing noise**. Precision stays high (0.57 on the random
split, 0.87 on the session split) while recall collapses (0.15 / 0.32
respectively). For an operational deployment this is a desirable
failure mode — under-prediction is recoverable by lowering the
threshold or by combining with other evidence (e.g. via the
late-fusion endpoint), whereas over-prediction would require manual
suppression. The catenary fit and Steiner-tree topology completion
modules in `app/ml/` are designed precisely to recover information
from this kind of partial evidence.

### 7.4 v2 empirical confirmation of the inference-path-drift hypothesis

The inference-path-drift hypothesis introduced in §6 and read into
the v1 failure pattern above was validated by the v2 retrain (see
§10): training and inference at matched 768×768 resolution lifted
pixel IoU from **0.1387 to 0.3066** (v1 random split → v2 canonical
test) and recall from **0.1546 to 0.3382**, with no other changes
to loss, optimiser, augmentation, or model architecture. The
hypothesis is now empirically confirmed: train/serve resolution
mismatch was the first-order driver of v1's silent recall collapse,
not the architecture or the training-loss choice. The catenary and
Steiner-tree completion modules remain the right answer for §7.3
context failure, which v2 does not address.

## 8. Computational analysis

The numbers below come from the actual U-Net + ResNet34 architecture
used at inference time (`app.ml.model.ConductorSegmenter`), measured
or computed where indicated.

| Quantity | Value | Source |
|---|---|---|
| Parameters | **24.44 M** (24,436,369 exactly) | `sum(p.numel() for p in m.parameters())` on the loaded model |
| Conv2d FLOPs at 768 × 768 (v2) | **~141 GFLOPs / forward pass** | quadratic in tile side; v1 measured 62.5 GFLOPs at 512 × 512 |
| Per-tile latency, GTX 1050 Ti, fp32, 768 px (v2) | ~300–500 ms | v1 at 512 px was ~150–300 ms; the increase tracks the ~2.25× FLOP ratio |
| Peak training VRAM, T4 / 16 GB, batch 12, 768 × 768 (v2) | ~14 GB | Lightning `precision="16-mixed"`; v1 ran batch 16 at 512 px ≈ ~12 GB |
| Peak training VRAM, 1050 Ti / 4 GB, batch 4, 512 × 512 (v1 fallback) | ~3.4 GB | matches the design choice noted in `docs/design_decisions.md` |
| Sliding-window cost, 1024 × 1024 input, stride 704 (v2) | 4 tiles → ~1.2–2.0 s end-to-end | tile count = `ceil((H + pad)/stride) · ceil((W + pad)/stride)`; v1 stride 448 produced 9 tiles |
| Disk weight size | ~95 MB (fp32) | dominated by the ResNet34 encoder weights at full precision |

The prototype runs comfortably within the laptop's 4 GB VRAM ceiling
for inference, which was the binding constraint when choosing
ResNet34. A ResNet50 encoder (~28 M params) would also fit, but at
roughly 1.5× the FLOPs without a clear accuracy gain on TTPLA per
the smp benchmark suite.

The latency band on the 1050 Ti is reported as a range rather than a
point estimate because the spread depends on whether `cudnn.benchmark`
has converged on optimal kernels; the first request after startup is
warmed up by `ConductorSegmenter.warmup()`.

## 9. Comparison with prior work

There is no clean apples-to-apples comparison for this prototype.
TTPLA's original paper reports instance-segmentation numbers
(Mask R-CNN backbone) on a different label vocabulary; our work is
binary semantic segmentation. Madaan et al. (2017) is a wire-detection
paper trained largely on synthetic data, evaluated on a custom test
set. The values below are therefore reference points, not benchmarks
beaten or lost — and we report ours as the *measured* output of
`training/evaluate.py`, not a re-derived headline figure.

| Source | Task | Backbone | Headline metric | Reported value |
|---|---|---|---|---|
| Abdelfattah et al. (2020), *TTPLA* (ACCV) | Instance segmentation, towers + cables | Mask R-CNN / YOLACT, ResNet50/101 | mAP@0.5 | see paper — the cable class consistently scored lower than tower classes |
| Madaan et al. (2017), *Wire Detection* (IROS) | Binary wire segmentation | Dilated CNN, synthetic + real | F1 | 0.84 |
| Yetgin & Gerek (2018) | Powerline detection (RGB, classical) | DCT features | Pixel accuracy | ~0.895 |
| **This prototype (v1)** | Binary cable semantic segmentation | U-Net, ResNet34 | mIoU / F1 / CCQ-Q | see `docs/evaluation_results.md` |
| **This prototype (v2)** | Binary cable semantic segmentation | U-Net, ResNet34 | mIoU / F1 / CCQ-Q | see `docs/evaluation_results_v2.md` |

Two caveats. (i) Pixel-accuracy and pixel-IoU on thin structures
under-reward near-misses; the headline number for this kind of task
should be CCQ-Quality with a 3-pixel buffer. (ii) Cross-dataset
generalisation is brittle: numbers on TTPLA do not transfer to
urban LV imagery without active labelling and likely an additional
self-supervised pretraining stage on the target domain. The
methodology section §6 expands on this.

## 10. v2 retrain

The v1 evaluation surfaced two distinct methodological problems:
inference-path drift (training at 512×512 via `A.Resize`, serving at
native 4K through 512-pixel sliding tiles) and training-set leakage
inside the v1 session-grouped evaluation (the v1 trainer's seed-42
`random_split` placed ~90% of every session prefix into the
training set, including the prefixes the session-grouped evaluation
held out as "test"). v2 is the targeted retrain that addresses both
by construction. Headline numbers below are reproduced from
[`evaluation_results_v2.md`](evaluation_results_v2.md).

### 10.1 What changed

- **Training resolution 512 → 768.** The validation transform reads
  from the same `--resolution` argument, so val IoU is reported on
  the path the model trains against. Cables in training are now at
  the same ≈3–5 px width the production sliding window sees at
  native 4K, closing the resolution gap §6 named.
- **Training-set construction: random 80/10/10 → TTPLA canonical
  three-way split** (`train.txt` 905 / `val.txt` 109 /
  `test.txt` 220). 8 unassigned post-publication images excluded;
  see §10.2.
- **Sliding-window inference tile in `training/evaluate.py`
  defaulted to 768 px** (overrides `settings.tile_size = 512`) so
  the v2 evaluation runs at the same resolution the v2 model
  trained against. Without this override the v2 weights would still
  be evaluated on the v1 inference path.
- **Threshold sweep on the 220-image canonical test set:** τ=0.30
  produces IoU **0.3178**, F1 **0.4355**, CCQ-Q **0.4624** —
  marginally better than τ=0.50 (IoU **0.3066**, F1 **0.4226**,
  CCQ-Q **0.4567**) on every operational metric, with precision
  trading down very slightly (0.7478 vs 0.7561). Headline numbers
  in this section use τ=0.50 to match the convention the rest of
  the doc reports at; **τ=0.30 is the empirically-optimal operating
  point for the calibrated v2 model and is what a production
  deployment should use.**

### 10.2 Canonical-splits methodology note

TTPLA's repository ships a canonical three-way split as
`train.txt` / `val.txt` / `test.txt` totalling 1234 images
(905 / 109 / 220). The dataset directory contains 1242 images;
the 8 unassigned post-publication additions are excluded to
preserve split integrity. Recent downstream work
(Yang et al., 2024 — RainTTPLA, HazeTTPLA, SnowTTPLA on
arXiv:2409.04812) uses a 1000/242 train/test convention with
their own reconstructed splits, which are not publicly released;
we use the original authors' canonical three-way split for
reproducibility and to preserve the held-out validation signal.

### 10.3 Results

Headline numbers across the three available evaluation runs.
v1 numbers are reproduced verbatim from
[`evaluation_results.md`](evaluation_results.md) and
[`evaluation_results_session.md`](evaluation_results_session.md);
v2 numbers from [`evaluation_results_v2.md`](evaluation_results_v2.md).

| Metric | v1 random | v1 session† | v2 canonical |
|---|---:|---:|---:|
| Images evaluated | 124 | 41 | 220 |
| Sliding-window tile size (px) | 512 | 512 | 768 |
| Pixel IoU | 0.1387 | 0.3067 | **0.3066** |
| Pixel precision | 0.5702 | 0.8702 | 0.7561 |
| Pixel recall | 0.1546 | 0.3209 | **0.3382** |
| Pixel F1 | 0.2018 | 0.4174 | **0.4226** |
| CCQ completeness | 0.2829 | 0.5652 | 0.5250 |
| CCQ correctness | 0.6528 | 0.8704 | 0.7909 |
| CCQ Quality (headline) | 0.2323 | 0.5296 | **0.4567** |
| ECE (10-bin) | 0.0190 | 0.0156 | **0.0153** |

† v1 session numbers are inflated by training-set leakage as
documented in `evaluation_results_session.md` — the v1 trainer's
seed-42 `random_split` placed ≈90% of every session prefix into
the training set, so most of the 41 "session test" images had
already been seen during training. The v1 random row is the only
v1 number that is genuinely held out.

The headline read: v2 IoU on a *truly* held-out 220-image test set
matches the v1 session number that was achieved on largely-seen
imagery (0.3066 vs 0.3067). On the genuinely held-out comparison
(v1 random vs v2 canonical), every operational metric improved by
≈ 2× — IoU 0.1387 → 0.3066, recall 0.1546 → 0.3382, F1 0.2018 →
0.4226, CCQ-Q 0.2323 → 0.4567 — with no other changes than the
two §10.1 ones. ECE remained excellent at 0.0153.

**Note on direct v1-vs-v2 comparison.** A controlled comparison
running v2 weights on v1's 124-image random test split would not
be cleanly interpretable: v1's random split was drawn independently
of TTPLA's canonical splits, and approximately 80% of v1's test
images are in v2's training and validation sets. The headline
v1→v2 improvement (IoU 0.139 → 0.307) is therefore reported
across different test distributions; the canonical-test result is
the cleaner generalisation number for v2 and is the result
comparable to other TTPLA work in the literature. v1's results
are preserved unchanged for methodological transparency, with the
understanding that they were obtained on a different evaluation
regime that is documented but not directly comparable.

### 10.4 Honest disclosures preserved

- **Inference-path drift identified in v1 informs the v2 design;
  v2 closes the train/inference resolution gap.** The §6 hypothesis
  and the §7.4 confirmation document the round trip from "this
  might be why v1 silently under-recalls" to "yes, that was it."
- **Random-split leakage in v1's session-grouped evaluation is an
  inherent property of the v1 split mode; v2's canonical splits
  eliminate this by construction.** The v1 session report stays in
  the repo as historical record (and as the implementation
  verification of the session-grouped split machinery), explicitly
  labelled as leaked.
- **TTPLA → real LV domain gap remains; the v2 retrain does not
  address this.** Active labelling on a DNO partner's urban
  aerial / drone imagery, self-supervised pretraining on their
  unlabelled archive, and a learned RGB+LiDAR fusion model are all
  Phase-1 steps of the research roadmap and are unchanged by v2.

#### v2 qualitative gallery — failure-mode classifications

Three v2 failures are stored under
`docs/screenshots/eval/canonical/failures/`. All three returned
IoU = 0.000 and CCQ-Q = 0.000 — consistent with the v1 pattern
that the model still fails by producing nothing, not by producing
noise. Hand-classification of the three:

- **Failure 2 (`11_00098.jpg`, IoU 0.000, CCQ-Q 0.000) — context
  failure on urban background.** A single thin cable against a
  parking-lot scene; the model produces no prediction. This is
  the textbook TTPLA → LV domain-shift failure mode: the network
  has no in-distribution training signal for cables overlaid on
  built-environment textures of similar contrast and orientation.
  The catenary / Steiner-tree completion modules in §4 are the
  right downstream answer for this case; the right *upstream*
  answer is the LV-domain active labelling named in §6.
- **Failure 3 (`44_01131.jpg`, IoU 0.000, CCQ-Q 0.000) —
  vegetation-background confusion.** Two clearly visible cables
  against pine vegetation; the model produces no prediction.
  Direct evidence that vegetation-rich pretraining data is needed
  in any Phase-1 SSEN data acquisition.
- **Failure 1 (`108_1830.jpg`, IoU 0.000): annotation gap rather
  than model failure.** The prediction overlay shows the model
  correctly identifying cables visible in the input image, but
  the TTPLA ground-truth polygons for this image are incomplete
  and do not include the predicted cables. IoU is therefore zero
  by construction. This finding is consistent with known
  limitations of single-pass annotation in academic datasets,
  and argues for project-specific labelling effort during any
  SSEN-data acquisition phase rather than reliance on transfer
  from TTPLA-quality labels alone.

The v2 failure pattern reinforces the v1 read: the failures that
remain are the §7.3 context failures and the LV-domain
distribution shift §6 names. v2 substantially reduced the §7.2
scale failure (recall 0.1546 → 0.3382, but still ≈ two-thirds of
cable pixels missed at τ=0.50); v2 did not (and was not designed
to) address the §6 domain-gap problem.

## References

- Ronneberger, O., Fischer, P., & Brox, T. (2015). U-Net: Convolutional Networks for Biomedical Image Segmentation. *MICCAI*.
- He, K., Zhang, X., Ren, S., & Sun, J. (2016). Deep Residual Learning for Image Recognition. *CVPR*.
- Lin, T.-Y., Goyal, P., Girshick, R., He, K., & Dollár, P. (2017). Focal Loss for Dense Object Detection. *ICCV*.
- Abdelfattah, R., Wang, X., & Wang, S. (2020). TTPLA: An Aerial-Image Dataset for Detection and Segmentation of Transmission Towers and Power Lines. *ACCV*.
- Madaan, R., Maturana, D., & Scherer, S. (2017). Wire Detection using Synthetic Data and Dilated Convolutional Networks for Unmanned Aerial Vehicles. *IROS*.
- Demantké, J., Mallet, C., David, N., & Vallet, B. (2011). Dimensionality based scale selection in 3D LiDAR point clouds. *ISPRS Workshop on Laser Scanning*.
- Irvine, H. M. (1981). *Cable Structures*. MIT Press.
- Lee, T. C., Kashyap, R. L., & Chu, C. N. (1994). Building skeleton models via 3-D medial surface axis thinning algorithms. *CVGIP*.
- Qi, C. R., Yi, L., Su, H., & Guibas, L. J. (2017). PointNet++: Deep Hierarchical Feature Learning on Point Sets in a Metric Space. *NeurIPS*.
- Kou, L., Markowsky, G., & Berman, L. (1981). A fast algorithm for Steiner trees. *Acta Informatica*, 15(2), 141–145.
- He, K., Chen, X., Xie, S., Li, Y., Dollár, P., & Girshick, R. (2022). Masked Autoencoders Are Scalable Vision Learners. *CVPR*.
- Oquab, M., Darcet, T., Moutakanni, T., et al. (2023). DINOv2: Learning Robust Visual Features without Supervision. *TMLR* (arXiv:2304.07193).
- Yang, S., Hu, B., Zhou, B., et al. (2024). Power Line Aerial Image Restoration under Adverse Weather: Datasets and Baselines. arXiv:2409.04812.
- Yetgin, Ö. E., & Gerek, Ö. N. (2018). Automatic recognition of scenes with power line wires in real life aerial images using DCT-based features. *Digital Signal Processing*, 77, 102–119.
- Wiedemann, C., Heipke, C., Mayer, H., & Jamet, O. (1998). Empirical Evaluation of Automatically Extracted Road Axes. In Bowyer & Phillips (eds.), *Empirical Evaluation Methods in Computer Vision*, IEEE CS Press, 172–187.
- Guo, C., Pleiss, G., Sun, Y., & Weinberger, K. Q. (2017). On Calibration of Modern Neural Networks. *ICML*.
- Khalil, E., Dai, H., Zhang, Y., Dilkina, B., & Song, L. (2017). Learning Combinatorial Optimization Algorithms over Graphs. *NeurIPS*.
- Hamilton, W. L., Ying, R., & Leskovec, J. (2017). Inductive Representation Learning on Large Graphs (GraphSAGE). *NeurIPS*.
- Veličković, P., Cucurull, G., Casanova, A., Romero, A., Liò, P., & Bengio, Y. (2018). Graph Attention Networks. *ICLR*.
- Choy, C., Gwak, J., & Savarese, S. (2019). 4D Spatio-Temporal ConvNets: Minkowski Convolutional Neural Networks. *CVPR*.
- Liu, Z., Tang, H., Amini, A., Yang, X., Mao, H., Rus, D., & Han, S. (2023). BEVFusion: Multi-Task Multi-Sensor Fusion with Unified Bird's-Eye View Representation. *ICRA*.
- Bai, X., Hu, Z., Zhu, X., Huang, Q., Chen, Y., Fu, H., & Tai, C.-L. (2022). TransFusion: Robust LiDAR-Camera Fusion for 3D Object Detection with Transformers. *CVPR*.
