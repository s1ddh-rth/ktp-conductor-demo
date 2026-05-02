# Design decisions

This document captures the alternatives considered for every major
technical decision in the project, the reason for the chosen option,
and what would change for a production deployment at SSEN scale.

The format throughout: a short comparison table, a paragraph or two of
discussion, and an explicit "what would change for production" note.

---

## 1. Segmentation model architecture

| Option | Params (24M base) | VRAM (512², bs 8) | Notes | Reference |
|---|---|---|---|---|
| **U-Net (ResNet34) — chosen** | ~24M | ~3.0 GB | Symmetric encoder–decoder, skip connections preserve thin-structure detail | Ronneberger et al. (2015) |
| DeepLabV3+ (ResNet50) | ~40M | ~3.8 GB | Atrous convolutions, strong on objects with multi-scale context | Chen et al. (2018) |
| SegFormer-B0 | ~3.7M | ~2.6 GB | Transformer encoder; needs more data for thin structures | Xie et al. (2021) |
| Mask2Former (Swin-T) | ~47M | OOM on 4 GB | Universal segmentation; overkill for binary cable mask | Cheng et al. (2022) |
| HRNet (W18) | ~9.6M | ~3.2 GB | Maintains high-resolution branches throughout; strong on thin objects | Wang et al. (2020) |
| YOLOv8-seg (n) | ~3.4M | ~2.0 GB | Real-time instance segmentation; loses fidelity on thin pixels | Ultralytics |

**Why U-Net.** Three reasons specific to this problem. First, thin-
structure segmentation (cables, vessels, road centrelines, retinal
arteries) is the original setting U-Net was designed for, and the skip
connections are exactly what conserve the few-pixel-wide detail we
care about. Second, the `segmentation-models-pytorch` (smp) library
makes encoder swaps trivial — if ResNet34 underperforms, ResNet50,
EfficientNetV2-S, or Swin can be substituted by changing one string.
Third, training stability is high; transformer-encoder segmenters need
more data and longer schedules to converge cleanly, neither of which
fits the 5-day budget.

**Why not SegFormer.** It would be a strong alternative with more
training data — its B0 variant has lower memory than U-Net+ResNet34
and is competitive on standard benchmarks. The reason against is
empirical: in our preliminary tests, ImageNet-pretrained CNN encoders
fine-tune on small segmentation datasets faster than transformer
encoders, which need either much larger pretraining corpora or
SegFormer-specific pretraining (MiT) to reach U-Net parity. For TTPLA's
~1100 images, the convolutional inductive bias (local connectivity,
translation equivariance) is the right prior.

**What would change for production.** A 30-month KTP would have access
to SSEN's unlabelled aerial archive — likely tens of thousands of
images. With that scale, a self-supervised pretraining stage (DINOv2 or
MAE) on the unlabelled data, followed by fine-tuning with SegFormer or
Mask2Former, would likely outperform U-Net by 2–5 IoU points. We would
also evaluate HRNet (Wang et al., 2020) which has shown strong results
on thin-structure tasks like vessel segmentation specifically because
it preserves high-resolution features throughout the network rather
than recovering them in a decoder.

---

## 2. Encoder backbone

| Option | Params | ImageNet top-1 | VRAM @ 512² bs 8 | Inference (1050 Ti) | Notes |
|---|---|---|---|---|---|
| **ResNet34 — chosen** | 21.3M | 73.3% | ~3.0 GB | ~180 ms / 512² | Reliable, well-pretrained |
| ResNet50 | 25.6M | 76.1% | ~3.8 GB | ~250 ms / 512² | Marginal gain, tight on 4 GB |
| EfficientNetV2-S | 21.5M | 83.9% | ~3.4 GB | ~220 ms / 512² | Best ImageNet but slower in practice |
| MobileNetV3-Large | 5.5M | 75.2% | ~2.0 GB | ~95 ms / 512² | Fast but capacity-limited for thin-structure |
| Swin-T | 28.3M | 81.3% | ~4.2 GB | OOM training | Good for large-context tasks |
| ConvNeXt-T | 28.6M | 82.1% | ~3.6 GB | ~210 ms / 512² | Modern conv design, marginally better than RN50 |

**Why ResNet34.** The 4 GB VRAM ceiling on the GTX 1050 Ti is the
binding constraint. ResNet34 trains comfortably at batch size 8, leaves
headroom for augmentation in-memory, and ImageNet weights are
exceptionally well-pretrained (it's been the most-used encoder for
seven years and remains the most thoroughly benchmarked encoder of
its generation). EfficientNetV2
gives a higher ImageNet score but its mobile-inverted-bottleneck blocks
are slower in practice on Pascal-generation GPUs without the optimised
TensorRT kernels. ConvNeXt-T scores higher on ImageNet but adds 7M
params
without commensurate IoU gain on segmentation when the dataset is
small.

**What would change for production.** With access to A100/H100 hardware
at SSEN, EfficientNetV2-L or ConvNeXt-Base would be obvious upgrades.
The bigger lever is self-supervised pretraining on SSEN's data: a
DINOv2-pretrained ViT-S/B encoder used in a U-Net-style decoder would
likely beat all of the above for the LV-specific task.

---

## 3. Loss function

| Option | Class-imbalance handling | Strength on thin structures | Stability |
|---|---|---|---|
| **0.5·Focal + 0.5·Dice — chosen** | Strong (focal) | Strong (Dice) | High |
| BCE | Weak | Weak | High |
| BCE + Dice | Moderate | Strong | High |
| Tversky | Tunable α/β | Moderate | Moderate |
| Focal Tversky | Strong, tunable | Strong | Moderate |
| Lovász-Softmax | Direct IoU optimisation | Strong | Lower (gradient noise) |
| Boundary loss (Kervadec) | Distance-aware | Very strong | Lower, needs warm-up |

**Why focal+Dice.** Cable segmentation is severely class-imbalanced —
typically <2% of pixels are conductor. BCE alone underweights the
positive class until a heavy `pos_weight` is applied, which makes
training unstable. Focal loss (Lin et al., 2017) handles this by
down-weighting easy negatives without the manual reweighting. Dice
loss complements focal by directly rewarding mask overlap rather than
per-pixel correctness, which is what we actually care about for thin
objects. The 50/50 mix is a standard recipe in the smp library and in
medical-imaging segmentation literature.

**Alternatives we considered and rejected.** Lovász-Softmax directly
optimises the Jaccard index but introduces gradient noise that slows
convergence — risky in a 2-hour training budget. Boundary loss
(Kervadec et al., 2019) is theoretically attractive for thin-structure
tasks because it penalises errors weighted by their distance from the
true boundary, but it requires a warm-up phase with a stable region
loss before introducing the boundary term, which we don't have time
for. Tversky/Focal-Tversky needs hyperparameter tuning of α/β which is
again tight given our wall-clock budget.

**What would change for production.** A KTP Phase 2 would test
boundary loss properly with the right warm-up schedule, and likely
include a topology-aware loss (CL-Dice or skeleton-recall loss, Shit
et al., 2021) which directly optimises the connectivity properties we
care about for downstream graph extraction.

---

## 4. Augmentation library

| Option | Maintained | Mask support | Speed | Ease of use |
|---|---|---|---|---|
| **Albumentations — chosen** | Yes (active) | Native, well-tested | Fast (NumPy/cv2) | Excellent |
| torchvision.transforms.v2 | Yes | Recently added | Fast (tensor) | Good |
| Kornia | Yes | Native | GPU-accelerated | Good |
| imgaug | Stale | Native | Slow | OK |
| Custom NumPy | n/a | n/a | Variable | Painful |

**Why Albumentations.** It is the de facto standard for segmentation
augmentation in research papers. Mask transforms are correctly
synchronised with image transforms (a non-trivial property — many
libraries rotate the image but not the mask, or apply colour jitter to
the mask). The probability semantics (`p=0.5` per transform plus
optional `OneOf` blocks) match how augmentation is reported in
methodology sections of papers.

**Why not torchvision.transforms.v2.** A close runner-up. The v2 API
finally supports tensor-mask synchronisation correctly, and the
performance is good. The reason for sticking with Albumentations is
that the mask-segmentation augmentation literature (and most
segmentation-models-pytorch examples) uses Albumentations idioms; this
keeps the recipe legible to anyone with segmentation background.

**What would change for production.** Likely nothing — Albumentations
remains the right choice. The interesting upgrade is moving to a
dataset-conditioned augmentation policy (e.g. AutoAugment-style
search) once the labelled-data scale supports it.

---

## 5. Training framework

| Option | Boilerplate | Lock-in risk | Distributed training | Custom callbacks |
|---|---|---|---|---|
| **PyTorch Lightning — chosen** | Low | Low (still raw nn.Module) | Excellent | Excellent |
| Vanilla PyTorch | High | None | Manual | Manual |
| HuggingFace Trainer | Low | High (ties to Transformers) | Excellent | Limited |
| MMSegmentation | Low | Very high (whole config DSL) | Excellent | Excellent (within DSL) |
| Composer (MosaicML) | Low | Moderate | Excellent | Excellent |
| Catalyst | Low | Moderate | Good | Excellent |

**Why Lightning.** Three properties matter for this project. First,
the model is still a plain `nn.Module` — Lightning is a training-loop
abstraction, not an architecture framework, so we're not locked in if
we later move to vanilla PyTorch or another runner. Second, the
checkpoint format is loadable with plain `torch.load`, which is what
the inference server (`app/ml/model.py`) does — no Lightning runtime
needed at serving time. Third, AMP, gradient clipping, LR scheduling,
multi-GPU, and best-checkpoint-by-metric are one-liners; we get them
for free.

**Why not MMSegmentation.** The most fully featured option for
segmentation specifically, but its config-DSL model means investing in
the framework's mental model rather than the code. For a 5-day sprint
with one developer, Lightning's "subclass and override" model is more
direct.

**Why not HuggingFace Trainer.** It's optimised for Transformers; the
ergonomics for non-Transformer segmentation are awkward. Different
problem, wrong tool.

**What would change for production.** For a 30-month project with
multi-GPU training and many architecture experiments,
MMSegmentation's config-driven approach genuinely speeds up
iteration. We would migrate to it after Phase 1.

---

## 6. Web framework

| Option | Async | Type safety | OpenAPI | Performance | Ecosystem |
|---|---|---|---|---|---|
| **FastAPI — chosen** | Yes | Pydantic-native | Free | Excellent | Large |
| Flask | Optional (async views) | Manual | Plugin | Good | Largest |
| Django | Yes (3.1+) | Manual | Plugin (DRF + drf-spectacular) | Good | Largest, batteries included |
| Litestar | Yes | Pydantic / msgspec | Free | Excellent | Smaller, growing |
| LitServe | Yes (LightningAI) | Manual | Limited | Excellent for ML | ML-specific |

**Why FastAPI.** Type-safety is the headline feature: Pydantic models
for request and response shapes catch a class of bugs at API
boundaries before they hit production. The free OpenAPI spec at `/docs`
means a user can interact with the API without
reading client code. Async support is genuine and the performance
overhead is negligible. The ecosystem is large enough that everything
we need (rate limiting, structured logging, Prometheus metrics) has a
maintained integration.

**Why not Litestar.** Litestar is a worthy competitor — arguably
better-designed in some areas — but its ecosystem is younger. For a
5-day sprint with a tight deadline, the maturity of FastAPI's
integrations is the deciding factor.

**Why not LitServe.** LitServe is purpose-built for ML inference
serving and has nice features for batching and GPU sharing. The reason
against is that the demo is more than just a model server — it has a
static frontend, multiple endpoints, GeoJSON computation, LiDAR
processing — and FastAPI is the right tool for "general API server",
not LitServe.

**What would change for production.** SSEN's existing Python services
likely use Flask or Django; alignment with their stack is a tiebreaker
worth respecting if it exists. FastAPI would still be a defensible
choice, but harmony with the host organisation matters.

---

## 7. Map and 3D viewer

### Map

| Option | License | Tile cost | Bundle size | Vector tiles |
|---|---|---|---|---|
| **Leaflet 1.9 — chosen** | BSD | Free OSM | Tiny (~40 KB) | Plugin only |
| Mapbox GL JS | Proprietary | Free below threshold, then paid | Larger (~250 KB) | Native |
| OpenLayers | BSD | Free OSM | Larger (~200 KB) | Native |
| MapLibre GL | BSD | Free OSM (with style server) | Larger (~250 KB) | Native |
| deck.gl | MIT | n/a (rendering only) | Large (~500 KB) | Designed for this |
| Cesium | Apache 2.0 | Free with attribution | Very large (~2 MB) | Native, 3D first |

**Why Leaflet.** The demo needs to render GeoJSON linestrings on a
basemap with no exotic styling. Leaflet does this in 40 KB with no
API token, no usage counter, no risk of the demo breaking on Monday
because we exceeded a free-tier quota. The OpenStreetMap tile server
allows non-commercial use without registration. For our needs, the
modern alternatives' advantages (vector tiles, GPU-accelerated
rendering) are unused.

**Why not Mapbox GL JS.** Beautiful and fast, but the license requires
an API token tied to an account; we don't want a single point of
failure. Free-tier limits also mean a viral demo could be throttled.

**Why not deck.gl.** Excellent for high-volume data visualization
overlays (millions of points). We have hundreds of linestrings;
overkill.

**What would change for production.** SSEN's GIS team uses ESRI
ArcGIS. The demo's map output (GeoJSON) is the right intermediate
format for ingesting into ArcGIS. For an internal SSEN tool, we'd use
the ArcGIS Maps SDK for JavaScript.

### 3D viewer

| Option | License | Bundle | Point cloud support | Ease |
|---|---|---|---|---|
| **Three.js — chosen** | MIT | ~600 KB | Via custom BufferGeometry | Excellent docs |
| Babylon.js | Apache 2.0 | ~1 MB | Native point clouds | Excellent docs |
| deck.gl PointCloudLayer | MIT | ~500 KB | Yes | Good |
| Potree | BSD | ~400 KB | Designed for it | Excellent for huge clouds |

**Why Three.js.** A few-hundred-thousand-point cloud is comfortably
within Three.js's capability via a `BufferGeometry` with per-vertex
colours. The library is the most-documented 3D framework on the web
and the community has more example code than any alternative. Bundle
size is acceptable served from a CDN.

**Why not Potree.** Designed for massive (hundreds of millions of
points) clouds with octree streaming. Wonderful tool, wrong scale for
our 150k-point demo.

---

## 8. LiDAR I/O and processing

| Option | LAS read | LAZ read | Speed | Pure Python |
|---|---|---|---|---|
| **laspy + lazrs — chosen** | Yes | Yes (via lazrs) | Good | Yes |
| PDAL Python bindings | Yes | Yes | Best | No (PDAL binary) |
| Open3D | Limited | Limited | Good | No (C++) |
| pylas (deprecated) | Yes | Yes | Good | Yes |

**Why laspy.** The demo only needs to read a few LAS/LAZ tiles. laspy
is pure-Python (with a small Rust dependency for LAZ via lazrs),
installs cleanly, and the API is simple. We don't need PDAL's
streaming-pipeline capability for this scale.

**What would change for production.** LV-scale operations would
involve county-level LiDAR (tens of GB per area). PDAL becomes
mandatory at that scale because of its streaming and tiling pipelines.

---

## 9. Graph algorithms

| Option | License | Pure Python | Steiner tree | Performance |
|---|---|---|---|---|
| **NetworkX — chosen** | BSD | Yes | `approximation.steiner_tree` | Adequate for <10k nodes |
| igraph | GPL | C core, Python wrapper | Yes | Better for >10k nodes |
| graph-tool | LGPL | C++ core | Yes | Best for large graphs |
| PyTorch Geometric | MIT | Yes | n/a (learning, not algos) | n/a |

**Why NetworkX.** The graphs in this demo are small (tens to hundreds
of nodes). NetworkX is pure Python with no compilation step, and its
Steiner-tree approximation (Kou-Markowsky-Berman) is exactly the
algorithm we need. graph-tool would be faster but adds a compiled
dependency.

**What would change for production.** For city-scale topology
inference (every building in a SSEN region), graph-tool or igraph
would replace NetworkX. The algorithm choice would also shift from
classical Steiner-tree approximation to a learned graph neural
network.

---

## 10. Dependency management

| Option | Speed | Lockfile | Editable installs | Workspace |
|---|---|---|---|---|
| **uv — chosen** | Fastest | Yes | Yes | Yes |
| Poetry | Slow | Yes | Yes | Limited |
| pip + venv | Slowest | requirements.txt only | Yes | n/a |
| conda / mamba | Slow | env.yml | Yes | n/a |
| pdm | Fast | Yes | Yes | Yes |

**Why uv.** It's roughly 10–100× faster than pip for the same
operation, has a proper lockfile, supports editable installs and
workspaces, and is from Astral (who also make ruff — we trust their
engineering). The speed matters: every fresh `uv sync` on the laptop
takes seconds, not minutes, which means iteration cost is low.

**Why not Poetry.** Poetry would also work. It's slower (the resolver
in particular) and the tooling is older. uv is the modern equivalent
with better defaults.

**Why not conda.** conda is appropriate for projects that need C
libraries outside of PyPI's reach. Everything we use is on PyPI;
conda's complexity would not pay back.

---

## 11. Deployment exposure

| Option | URL quality | HTTPS | Free tier | Reconnect window | Notes |
|---|---|---|---|---|---|
| **cloudflared (named) — chosen** | Custom domain | Auto | Yes | None | Production-grade |
| cloudflared (try) | random.trycloudflare.com | Auto | Yes | Per-session | Demo-quality URL |
| ngrok (free) | random.ngrok.io | Auto | Yes | 8h reconnect | Free tier hostile |
| Tailscale Funnel | ts.net subdomain | Auto | Yes (with Tailscale) | None | Requires Tailscale on client |
| Port forward + DDNS | Variable | Manual | Yes | None | Brittle, ISP-dependent |

**Why cloudflared.** The "named tunnel" mode binds a Cloudflare-managed
subdomain (e.g. `ktp.<yourdomain>.dev`) to a local service via an
authenticated tunnel. There is no port forwarding, no public IP
exposure, and HTTPS is automatic. Visitors see a professional URL on
a real domain. The only cost is owning a domain (one-time ~£10 on
Cloudflare Registrar).

**Why not ngrok.** The free tier disconnects every 8 hours, which is
unacceptable for a demo that needs to be live for 48–72 hours after
the email is sent.

**Why not port forwarding.** Brittle (depends on the ISP not blocking
inbound), insecure (exposes the laptop's address), and no HTTPS
without Let's Encrypt setup we don't have time for.

---

## 12. Compute split (training vs serving)

The split (Colab T4 for training, Fedora 1050 Ti for serving) is the
only choice in this list driven by time-budget rather than design
preference. The alternatives:

- **All on the laptop.** Saturday-evening training would take ~6
  hours on the 1050 Ti, then run out of thermal budget. Achievable
  but risky.
- **All on Colab.** Free Colab can serve via ngrok-style tunnels but
  the 12-hour session limit kills the demo before Monday is over.
- **Paid cloud.** Lambda Labs, Vast.ai, or AWS would solve everything
  for ~£20–50, but adds account-setup overhead and means the demo
  isn't actually running on the developer's hardware (worth
  discussing).

The chosen split gets the speed advantage of Colab T4 (~3× faster
training than the 1050 Ti and no thermal concerns) while keeping the
serving story honest — when a visitor opens the link, every byte
comes from the developer's laptop.

---

## 13. Frontend stack

| Choice | Alternative | Reason |
|---|---|---|
| **No build step** | Webpack/Vite + React | One developer, 5 days; build complexity adds zero user-visible value |
| **Tailwind via CDN** | Tailwind compiled, custom CSS | CDN is dramatically simpler; we're not bandwidth-sensitive |
| **Vanilla ES modules** | TypeScript + bundler | Keeps the surface area small; no debug-time confusion about source maps |
| **import-map for Three.js** | Bundle Three.js | The CDN ESM build works directly; bundling buys us nothing |
| **Leaflet + Three.js** | A unified GIS library (Cesium) | Two specialised libraries are simpler than one general one |

The principle: every dependency is a maintenance cost. For a 5-day
sprint with a single developer, the build step is the highest-ROI
thing to skip.

---

## Summary table — the whole stack at a glance

| Concern | Choice | Top alternative we'd reach for at scale |
|---|---|---|
| Segmentation model | U-Net / ResNet34 | SegFormer-B2 with self-supervised pretraining |
| Loss | Focal + Dice (50/50) | Focal + Dice + CL-Dice (topology-aware) |
| Augmentation | Albumentations | Albumentations + AutoAugment search |
| Training framework | PyTorch Lightning | MMSegmentation |
| Web framework | FastAPI | FastAPI (no change) |
| Map | Leaflet | ArcGIS Maps SDK (if SSEN-internal) |
| 3D viewer | Three.js | Potree (large clouds) |
| LiDAR I/O | laspy | PDAL |
| Graph algorithms | NetworkX | graph-tool + GNN (PyG) |
| Dep manager | uv | uv (no change) |
| Deployment | cloudflared named | Kubernetes behind Cloudflare Access |
| Build step | None | Vite + TypeScript + React (if frontend grows) |
