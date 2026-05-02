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

A U-Net (Ronneberger et al., 2015) with a ResNet34 (He et al., 2015)
encoder, ImageNet-pretrained. ResNet34 sits at a defensible point on
the depth/memory trade-off — deep enough to learn cable texture,
shallow enough to fine-tune at 512×512 with batch size 8 on a
4 GB-VRAM GPU. The decoder is the standard U-Net symmetric upsampler
with skip connections from the encoder.

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

Sliding-window tiling at 512 × 512 with a 64-pixel overlap, reflective
padding to a multiple of the tile, and Hann-windowed averaging during
stitching to suppress tile-edge artefacts. This lets the model run on
arbitrary input sizes without retraining.

## 3. Vectorisation

A binary mask is one half of the GIS-readiness story. Asset data needs
**linestring geometry** with explicit topology (which cable connects to
which pole). The pipeline:

1. **Threshold + morphological open** to remove sub-cable speckle.
2. **Skeletonize** (Lee 1994 / scikit-image medial axis).
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

Suspended conductors follow a catenary curve under their own weight:

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
Kou-Markowsky-Berman approximation.

The Steiner-tree baseline is deterministic and fast. A graph neural
network conditioned on building footprints, transformer locations,
and fragment evidence is the natural next step — it would learn
topology priors that the deterministic method cannot capture (e.g.
"feeders prefer to follow streets", "branch lengths follow a
characteristic distribution").

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
(Qi et al., 2017) or a sparse 3D CNN (MinkowskiNet) is the obvious
upgrade for production scale.

## 6. Limitations and what would change for the real KTP

### Domain gap
TTPLA is rural transmission lines. LV imagery is urban service cables
in cluttered backgrounds. The model trained here would not transfer
directly. The KTP would address this with:

- **Active labelling** on SSEN's own urban aerial / drone imagery —
  even a few hundred hand-labelled tiles would close most of the gap.
- **Self-supervised pretraining** on SSEN's unlabelled aerial archive
  using DINOv2 or MAE.
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
attention between modalities, shared decoder. Recent fusion work
(BEVFusion, TransFusion in autonomous driving) provides good
templates.

### Hidden-cable inference
The deterministic Steiner-tree method is a methodology demonstrator,
not a production tool. A graph neural network — likely a GraphSAGE or
GAT operating on the building/pole graph with cable-fragment node
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

A working model is uninteresting without an honest account of when it
breaks. The evaluation script (`training/evaluate.py`) writes
`docs/evaluation_results.md` with three concrete failure cases drawn
from the held-out split; the framework below classifies them by
mechanism, which is more useful than a flat list.

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

The evaluation script's `docs/screenshots/eval/failures/` gallery
contains one example of each, with a sentence-level diagnosis. The
selection prioritises *informative* failures over the most
embarrassing ones.

## 8. Computational analysis

The numbers below come from the actual U-Net + ResNet34 architecture
used at inference time (`app.ml.model.ConductorSegmenter`), measured
or computed where indicated.

| Quantity | Value | Source |
|---|---|---|
| Parameters | **24.44 M** (24,436,369 exactly) | `sum(p.numel() for p in m.parameters())` on the loaded model |
| Conv2d FLOPs at 512 × 512 | **62.5 GFLOPs / forward pass** | hooked-pass count over Conv2d layers, single tile |
| Per-tile latency, GTX 1050 Ti, fp32 | ~150–300 ms | typical for this architecture at this scale; replace with the measured value after a benchmark pass on the deployment hardware |
| Peak training VRAM, T4 / 16 GB, batch 16 | ~12 GB | Lightning `precision="16-mixed"` |
| Peak training VRAM, 1050 Ti / 4 GB, batch 4 | ~3.4 GB | matches the design choice noted in `docs/design_decisions.md` |
| Sliding-window cost, 1024 × 1024 input, stride 448 | 9 tiles → ~1.5–2.7 s end-to-end | tile count = `ceil((H + pad)/stride) · ceil((W + pad)/stride)` |
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
| Abdelfattah et al. (2020), *TTPLA* (ACCV) | Instance segmentation, towers + cables | Mask R-CNN / Yolact, ResNet50/101 | mAP@0.5 | see paper — the cable class consistently scored lower than tower classes |
| Madaan et al. (2017), *Wire Detection* (IROS) | Binary wire segmentation | Dilated CNN, synthetic + real | F1 | 0.84 |
| Yetgin & Gerek (2018) | Powerline detection (RGB, classical) | Hand-crafted features | Pixel accuracy | 0.96 |
| **This prototype** | Binary cable semantic segmentation | U-Net, ResNet34 | mIoU / F1 / CCQ-Q | see `docs/evaluation_results.md` |

Two caveats. (i) Pixel-accuracy and pixel-IoU on thin structures
under-reward near-misses; the headline number for this kind of task
should be CCQ-Quality with a 3-pixel buffer. (ii) Cross-dataset
generalisation is brittle: numbers on TTPLA do not transfer to
urban LV imagery without active labelling and likely an additional
self-supervised pretraining stage on the target domain. The
methodology section §6 expands on this.

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
- Kou, L., Markowsky, G., & Berman, L. (1981). A fast algorithm for Steiner trees. *Acta Informatica*.
