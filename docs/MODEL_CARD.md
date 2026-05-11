# Model card — `unet_resnet34_ttpla`

Following Mitchell et al. (2019), *Model Cards for Model Reporting*.
This card documents the segmentation model produced by
`training/train.py` and served by the FastAPI app at `/api/segment`,
`/api/vectorise`, and `/api/fuse`.

## Model details

| | |
|---|---|
| **Architecture** | U-Net (Ronneberger et al., 2015) with ResNet-34 encoder (He et al., 2016), ImageNet-pretrained, single-channel binary output. |
| **Framework** | PyTorch 2.4 + `segmentation-models-pytorch` 0.3.4. |
| **Parameters** | 24,436,369 trainable parameters; ~95 MB on disk in fp32. |
| **Input** | RGB image, ImageNet-normalised, sliding 512 × 512 tiles with 64-pixel overlap and Hann-windowed stitching for arbitrary input sizes. |
| **Output** | Per-pixel probability of the "conductor" class in [0, 1]; thresholded at 0.5 by default. |
| **Loss** | 0.5 × FocalLoss (Lin et al., 2017) + 0.5 × DiceLoss. |
| **Optimiser** | AdamW, lr = 1e-3, weight decay = 1e-4, cosine annealing. |
| **Training schedule** | Up to 50 epochs, batch size 16, image crops 512 × 512, mixed precision. |
| **Producer** | This repository's `training/train.py`. |
| **Licence** | Same as the repository (MIT). Trained on Apache-2.0 TTPLA data. |

## Intended use

Designed for **research and methodology demonstration** of a conductor-
mapping pipeline against the wider context of UK low-voltage network
digitisation. The model takes RGB aerial imagery and outputs a binary
"cable / not-cable" mask, intended as input to a downstream
vectorisation pipeline that produces GIS-ready GeoJSON.

**Downstream uses** within this repository:

- `app/ml/postprocess.py` — mask → skeleton → graph → simplified
  LineStrings.
- `app/geo/vectorise.py` — LineStrings → GeoJSON FeatureCollection.
- `app/ml/fusion.py` — late-fusion re-scoring with airborne LiDAR.
- `app/ml/catenary.py`, `app/ml/graph_complete.py` — hidden-cable
  inference downstream of partial RGB evidence.

## Out-of-scope use

The model is **not** validated for:

- Operational asset registration in any DNO production system.
- Real-time safety decisions (e.g. dispatch routing, switching).
- Imagery distinct from the TTPLA training distribution, including
  but not limited to: urban LV cabling, low-altitude drone footage
  outside the capture parameters of TTPLA, and infrared / multispectral
  bands.

## Training data

| | |
|---|---|
| **Source** | TTPLA dataset (Abdelfattah, Wang, Wang. *TTPLA: An Aerial-Image Dataset for Detection and Segmentation of Transmission Towers and Power Lines.* ACCV 2020). |
| **Licence** | Apache 2.0. |
| **Size** | 1,242 (image, mask) pairs after polygon-to-mask conversion. |
| **Image format** | 3840 × 2160 JPEG. |
| **Subject** | Rural / inter-urban high-voltage transmission lines, captured by aerial flight. |
| **Train / val split as used** | Random 90 / 10 (`torch.utils.data.random_split`, seed 42). The trainer ignores TTPLA's canonical `splitting_dataset_txt/` partition — see Limitations below. |
| **Class taxonomy** | 5 classes in the source (`cable`, `tower_lattice`, `tower_wooden`, `tower_tucohy`, `void`); only `cable` is used as the positive class for this binary segmenter. |

## Evaluation

Numbers reported in `docs/evaluation_results.md` once `training/evaluate.py`
has been run against the deployed weights. The evaluation framework
(see `docs/evaluation.md`) covers:

- **Pixel IoU / precision / recall / F1** at threshold 0.5.
- **CCQ (Completeness, Correctness, Quality)** at a 3-pixel buffer
  tolerance, after Wiedemann, Heipke, Mayer, Jamet (1998).
- **Expected Calibration Error** (Guo et al., 2017), 10-bin reliability.
- A curated 3-success / 3-failure qualitative gallery, with failure
  diagnoses categorised as texture confusion, scale failure, or
  context failure (per `docs/methodology.md` §7).

## Known failure modes

Documented in `docs/methodology.md` §7. Summarised here:

- **Texture confusion** — fence shadows, road markings, or roof seams
  with sub-cable contrast and similar locally-linear geometry. The
  segmenter has insufficient receptive field to disambiguate purely
  from context.
- **Scale failure** — cables noticeably thicker or thinner than ~2–4
  pixels (TTPLA's typical regime). The model under-segments thick
  cables and misses sub-pixel ones.
- **Context failure** — cables visible to a human only because of
  surrounding structure (poles at known endpoints, with the visible
  cable signal vanishing midspan). The catenary and graph-completion
  modules downstream of the segmenter address this case explicitly.

## Limitations and caveats

1. **Domain gap to LV.** The model is trained on rural transmission
   lines; the application context (urban LV networks served by UK
   DNOs) is materially different in cable thickness, background
   distribution, and lighting. Performance on LV imagery is expected
   to be substantially worse without active labelling or self-
   supervised pretraining on a DNO's own archive.
2. **Random split methodology.** The trainer ignores TTPLA's canonical
   splits. Random splitting of an aerial dataset where same-flight
   images are highly correlated tends to over-estimate generalisation.
   Run `training/evaluate.py --split-strategy session` for a stricter
   cross-flight evaluation.
3. **Single modality at training time.** The model sees no LiDAR. The
   `fuse` endpoint adds LiDAR signal at inference time as a late-fusion
   re-scoring step; a learned mid-fusion model is named as Phase-2 in
   `docs/research_roadmap.md`.
4. **No guarantee of geo-referencing accuracy.** The pixel-to-lat/lon
   transform in `app/geo/vectorise.py` maps onto a synthetic box near
   Thatcham for visualisation only; it does not consume camera-pose
   metadata.
5. **Uncertainty quantification.** The model outputs scalar
   probabilities only. ECE indicates whether those probabilities are
   well-calibrated; no proper Bayesian uncertainty (e.g. dropout-MC,
   ensemble) is computed.

## Carbon and compute footprint

- **Training**: ~25–50 minutes on a single Colab T4 GPU at fp16-mixed
  precision; rough peak power 70 W → 0.04–0.06 kWh per training run.
- **Inference**: ~150–300 ms per 512 × 512 tile on a GTX 1050 Ti;
  CPU inference (the demo deployment fallback) is ~5 seconds per tile.

## Citation

If you reference this prototype, please also cite:

```
@inproceedings{abdelfattah2020ttpla,
  title     = {TTPLA: An Aerial-Image Dataset for Detection and Segmentation of Transmission Towers and Power Lines},
  author    = {Abdelfattah, Rabab and Wang, Xiaofeng and Wang, Song},
  booktitle = {Proceedings of the Asian Conference on Computer Vision (ACCV)},
  year      = {2020},
}

@inproceedings{ronneberger2015unet,
  title     = {U-Net: Convolutional Networks for Biomedical Image Segmentation},
  author    = {Ronneberger, Olaf and Fischer, Philipp and Brox, Thomas},
  booktitle = {Medical Image Computing and Computer-Assisted Intervention (MICCAI)},
  year      = {2015},
}
```

## v2 model card — `unet_resnet34_ttpla_v2`

The v2 weights at `weights/unet_resnet34_ttpla_v2.pth` are the
current deployed model. The v1 sections above are preserved as
historical record; differences from v1 are documented in
`docs/methodology.md` §10. v2 architecture, loss, optimiser, and
augmentation are identical to v1; only the training resolution
and split protocol changed.

### v2 model details (delta from v1)

| | |
|---|---|
| **Model version** | v2 |
| **Training resolution** | 768 × 768 (v1: 512 × 512) |
| **Training schedule** | Up to 50 epochs, batch size 12, mixed precision (v1: batch size 16). |
| **Inference tile size** | 768 × 768 sliding window with 64-pixel overlap and Hann-windowed stitching (v1: 512 × 512). Matches the training resolution to close the inference-path drift documented in v1's evaluation. |
| **Producer** | `training/train.py` with `--split-mode canonical --resolution 768 --batch-size 12`. |

### v2 training data

| | |
|---|---|
| **Source** | TTPLA dataset (Abdelfattah et al. 2020), official canonical splits from `splitting_dataset_txt/`. |
| **Train / val / test split** | 905 / 109 / 220 from `train.txt` / `val.txt` / `test.txt`; total 1,234 images. The 8 unassigned post-publication images in the 1,242-image directory are excluded to preserve split integrity. |
| **Class taxonomy** | Unchanged from v1: only `cable` is used as the positive class. |

### v2 evaluation

Full report: `docs/evaluation_results_v2.md`. Numbers below are on
the 220-image canonical test split, sliding-window inference at
768 px, CCQ buffer 3 px.

| Metric | τ=0.50 (convention) | τ=0.30 (operational) |
|---|---:|---:|
| Pixel IoU | 0.307 | 0.318 |
| Pixel F1 | 0.423 | 0.436 |
| CCQ Quality | 0.457 | 0.462 |
| ECE (10-bin) | 0.015 | 0.015 |

**Operational threshold.** τ=0.30 is the empirically-optimal
operating point of the calibrated v2 model on the canonical test
set — marginally better on every operational metric than the
convention threshold τ=0.50, with precision trading down very
slightly (0.7478 vs 0.7561). ECE is unchanged because it is a
property of the probability map, not the threshold.

### v2 known failure modes

Hand-classified from the three IoU=0 failures in
`docs/screenshots/eval/canonical/failures/` (see
`docs/methodology.md` §10.4 for full diagnoses):

- **TTPLA → urban-LV domain shift** (`11_00098.jpg`) — single
  thin cable against a parking-lot scene; the model produces no
  prediction. This is the textbook domain-gap failure mode.
- **Vegetation-background confusion** (`44_01131.jpg`) — two
  clearly visible cables against pine vegetation; the model
  produces no prediction. Argues for vegetation-rich pretraining
  data in any Phase-1 SSEN data acquisition.
- **Annotation gap** (`108_1830.jpg`) — the prediction overlay
  shows the model correctly identifying cables visible in the
  input, but the TTPLA ground-truth polygons for this image are
  incomplete. IoU is zero by construction, not by model failure.

The v1 failure-mode categories (texture / scale / context, §7)
remain the right taxonomy. v2's empirical pattern: the §7.2 scale
failure was substantially reduced by the 768-px retrain; what
remains are §7.3 context failures and §6 domain-shift cases.

### v2 carbon and compute footprint

- **Training**: ~3 hours on a single Colab T4 GPU at fp16-mixed
  precision; rough peak power 70 W → ~0.5 kWh per training run
  (estimated GPU-time only; excludes CPU/RAM overhead).
- **Inference**: ~300–500 ms per 768 × 768 tile on a GTX 1050 Ti
  (≈1.5–2× the v1 512-tile latency); CPU inference is ~12 s
  per tile.

## Card history

| Version | Date | Changes |
|---|---|---|
| 0.1 | 2026-05 | Initial card. Architecture, training schedule, intended use, limitations, evaluation framework. |
| 0.2 | 2026-05 | v2 model card section added. Training at 768×768 with TTPLA canonical splits (905 / 109 / 220); evaluation on 220-image canonical test (IoU 0.307, F1 0.423, CCQ-Q 0.457, ECE 0.015 at τ=0.50). Failure modes hand-classified. v1 sections preserved unchanged. |
