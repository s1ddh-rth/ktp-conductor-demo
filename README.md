# LV Conductor Mapping — Prototype

RGB conductor segmentation is the core contribution. The hidden-cable,
LiDAR, and fusion modules are deterministic baselines that establish
performance floors and demonstrate the multi-modal pipeline a
production system requires; their learned successors are named in the
30-month research roadmap.

An end-to-end research prototype for mapping low-voltage (LV)
conductor networks from aerial imagery, motivated by the data-
digitisation challenges UK Distribution Network Operators face on
their LV estate.

The system runs end-to-end from an aerial RGB image to GIS-ready
vector geometry, with two further modules covering hidden-cable
inference and LiDAR classification.

## On the dataset gap, up front

LV-specific labelled imagery is not openly available, so the
segmentation model here is trained on **TTPLA** (Abdelfattah et al.,
2020) — high-voltage transmission lines in rural settings. That is
the wrong distribution for urban LV cables, and the prototype owns
the consequence rather than dressing over it:

- A **session-grouped evaluation split** (TTPLA filenames encode
  flight sessions) is reported alongside the random split, exposing
  generalisation gaps that pure random splitting would hide. See
  `docs/evaluation.md` and `training/evaluate.py`.
- A small **synthetic urban-LV scene generator**
  (`scripts/synthesise_lv.py`) produces deterministic procedural
  scenes the segmenter can be inspected on; partial recovery is
  expected and informative, not a failure.
- The methodology document is explicit about what would change with
  real LV data: active labelling, self-supervised pretraining on a
  DNO's unlabelled archive, and a learned RGB+LiDAR fusion model
  rather than the late-fusion baseline shipped here.

This framing is more useful as a research artefact than chasing a
single headline IoU on TTPLA.

## Headline results

The current model (v2) is a TTPLA-trained U-Net (ResNet34 encoder)
evaluated against the dataset authors' canonical 220-image test
split, with sliding-window inference at 768 px (the same resolution
the model trains at). Full report:
[`docs/evaluation_results_v2.md`](docs/evaluation_results_v2.md).

| Metric | Value |
|---|---:|
| Pixel IoU | **0.307** |
| Pixel F1 | **0.423** |
| CCQ Quality (3-px buffer) | **0.457** |
| Expected Calibration Error (10-bin) | **0.015** |

Numbers above are at the convention threshold τ=0.50. The threshold
sweep in the v2 report finds that **τ=0.30 is marginally better on
every operational metric** (IoU 0.318, F1 0.436, CCQ-Q 0.462) — that
is the production-recommended operating point for the calibrated
model. ECE is unchanged because it's a property of the probability
map, not the threshold.

The story behind these numbers — why v1 reported IoU 0.139, why the
session-grouped split returned IoU 0.307 but was leaked, and why v2's
matched-resolution retrain on the canonical split is the methodology
fix that closes both gaps — is told in
[`docs/methodology.md`](docs/methodology.md) §10.

### v1 historical results

v1 numbers are preserved in
[`docs/evaluation_results.md`](docs/evaluation_results.md) (random
split) and
[`docs/evaluation_results_session.md`](docs/evaluation_results_session.md)
(session-grouped split). They are kept as the methodology-evolution
narrative — both files document the v1 limitations (inference-path
drift at 512 px sliding window, training-set leakage in the
session-grouped evaluation) that the v2 retrain was designed to
address. They are not deleted because the round trip from "this
might be why v1 silently under-recalls" to v2's empirical
confirmation is itself the research finding the methodology doc
turns on.

| Metric | v1 random | v1 session† | v2 canonical |
|---|---:|---:|---:|
| Images evaluated | 124 | 41 | 220 |
| Pixel IoU | 0.1387 | 0.3067 | **0.3066** |
| Pixel F1 | 0.2018 | 0.4174 | **0.4226** |
| CCQ Quality | 0.2323 | 0.5296 | **0.4567** |
| ECE (10-bin) | 0.0190 | 0.0156 | **0.0153** |

† v1 session numbers are inflated by training-set leakage as
documented in `docs/evaluation_results_session.md`. The v1 random
row is the only v1 number that is genuinely held out and is the
correct comparison point for v2.

## What's in here

| Module | What it does | Where |
|---|---|---|
| RGB segmentation | U-Net (ResNet34, ImageNet-pretrained) trained on TTPLA, sliding-window inference for arbitrary image sizes | `app/ml/model.py`, `training/train.py` |
| Mask → graph | Skeletonisation, junction/endpoint detection, edge tracing | `app/ml/postprocess.py` |
| Vectorisation | LineStrings → simplified GeoJSON | `app/geo/vectorise.py` |
| Hidden-cable inference (catenary) | Physical curve fit between two visible anchors | `app/ml/catenary.py` |
| Hidden-cable inference (topology) | Steiner-tree approximation over building, pole, transformer locations, biased by observed fragments | `app/ml/graph_complete.py` |
| LiDAR classification | Eigenvalue features (Demantké et al., 2011) + height-above-ground | `app/geo/lidar_features.py` |
| Web app | FastAPI + vanilla JS + Leaflet + Three.js | `app/main.py`, `app/static/` |

## Run it

```bash
# Install (uv recommended)
uv sync

# Place trained weights (see training/COLAB.md)
mkdir -p weights
cp /path/to/unet_resnet34_ttpla_v2.pth weights/

# Add a sample LAS for the LiDAR demo
cp /path/to/sample.laz app/static/examples/thatcham_sample.laz

# Run
uv run uvicorn app.main:app --host 0.0.0.0 --port 8000
```

Then open `http://localhost:8000` in a browser.

### Run via Docker

```bash
docker compose up --build
# → http://localhost:8000
```

The provided `Dockerfile` and `docker-compose.yml` use the CUDA 12.4
runtime base; on a host with `nvidia-container-toolkit` they pick up
the GPU automatically. CPU-only hosts work too — `ConductorSegmenter`
falls back when CUDA is unavailable.

### Run via Podman on Fedora

The Dockerfile is plain syntax (no `# syntax=` BuildKit directive),
so Podman / Buildah parse it natively. The compose file's bind mounts
carry `:Z` so SELinux on Fedora doesn't block them. One known caveat:
podman-compose does not reliably honour the Compose v3 GPU
reservation block — the portable Podman GPU path is the
Container Device Interface (CDI), wired up once on the host:

```bash
# One-off host setup (Fedora 40+):
sudo dnf install -y nvidia-container-toolkit
sudo nvidia-ctk cdi generate --output=/etc/cdi/nvidia.yaml
nvidia-ctk cdi list  # should show nvidia.com/gpu=all and per-GPU entries
```

Build the image, then run with the GPU passed in via CDI:

```bash
podman build -t ktp-conductor-demo:latest .

podman run --rm -d \
    --name ktp-demo \
    --device nvidia.com/gpu=all \
    -p 8000:8000 \
    -e MODEL_WEIGHTS=weights/unet_resnet34_ttpla_v2.pth \
    -e TILE_SIZE=768 \
    -v "$PWD/weights:/app/weights:ro,Z" \
    -v "$PWD/app/static/examples:/app/app/static/examples:ro,Z" \
    ktp-conductor-demo:latest
```

`MODEL_WEIGHTS` and `TILE_SIZE=768` point the server at the v2
checkpoint and match the production sliding-window tile to the v2
training resolution. Omit `--device nvidia.com/gpu=all` and the
container runs on CPU silently — `torch.cuda.is_available()` simply
returns `False` and `ConductorSegmenter` falls back. On a GTX 1050 Ti
(4 GB), v2 inference at 768 × 768 takes ~300–500 ms per tile; the
CPU fallback is ~12 s per tile.

Confirm the GPU is visible inside the container:

```bash
podman exec ktp-demo python -c \
    "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
# → True NVIDIA GeForce GTX 1050 Ti
```

For a long-lived service, prefer a Podman **Quadlet** unit
(`/etc/containers/systemd/ktp-demo.container`) over the compose
`restart: unless-stopped` block — the Quadlet is what `systemd`
manages directly and is the idiomatic Fedora pattern. The
`scripts/ktp-demo.service` file in this repo is the bare-uvicorn
form; a Quadlet wrapping the container above is the equivalent for
the containerised deployment.

### Run as a systemd service

A unit template is at `scripts/ktp-demo.service`. Replace `__USER__`
with the actual host user, then:

```bash
sudo cp scripts/ktp-demo.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now ktp-demo
journalctl -u ktp-demo -f
```

`Restart=on-failure` brings the service back within ~10 seconds of
an unexpected exit, which matters for any long-running deployment.

### Health monitoring

`scripts/healthcheck.sh` curls `/health` and writes a one-line log
entry. Wire it into cron at 5-minute cadence:

```cron
*/5 * * * * /home/__USER__/ktp-conductor-demo/scripts/healthcheck.sh
```

### Synthetic LV examples

If you want UI-ready demo images without downloading TTPLA:

```bash
uv run python -m scripts.synthesise_lv --out app/static/examples --n 3
```

This produces three deterministic urban-LV-style scenes plus matching
ground-truth masks. They're listed in `app/static/examples/index.json`
so the frontend's "Examples" panel surfaces them automatically.

### Tests and evaluation

```bash
uv run python -m pytest             # 45 tests on ml/, geo/, routers
uv run python -m training.evaluate \
    --data /path/to/ttpla \
    --weights weights/unet_resnet34_ttpla_v2.pth \
    --split-strategy canonical --split test
# → docs/evaluation_results_v2.md + qualitative panels under docs/screenshots/eval/
```

## Stack

- **Python 3.11+**, dependencies pinned in `uv.lock`
- **PyTorch 2.4** + `segmentation-models-pytorch`
- **FastAPI** with rate limiting, CORS, structured logging, Prometheus metrics
- **Leaflet 1.9** for map rendering, **Three.js 0.160** for the 3D point cloud
- **scikit-image, NetworkX, Shapely, SciPy** for the postprocessing pipeline
- **laspy** for LAS/LAZ reading

## Methodology and limitations

Read `docs/methodology.md` for the discussion of:

- Why TTPLA was chosen as a training proxy for unavailable LV-specific data
- The domain gap between transmission and distribution imagery
- Limitations of the catenary fit and Steiner-tree topology approaches
- What a full Knowledge Transfer Partnership (KTP) project would do differently with real LV data

## Acknowledgements

Trained on the TTPLA dataset (Abdelfattah et al., 2020). LiDAR classification
follows the eigenvalue-feature scheme of Demantké et al. (2011). The catenary
fit follows Irvine (1981). The U-Net architecture is Ronneberger et al. (2015).

Built as a prototype demonstration.

## Licence

MIT.
