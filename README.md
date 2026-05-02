# LV Conductor Mapping — Prototype

An end-to-end research prototype for mapping low-voltage (LV)
conductor networks from aerial imagery, motivated by the data-
digitisation challenges UK Distribution Network Operators face on
their LV estate.

The system runs end-to-end from an aerial RGB image to GIS-ready
vector geometry, with two further modules covering hidden-cable
inference and LiDAR classification.

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
cp /path/to/unet_resnet34_ttpla.pth weights/

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
uv run python -m pytest             # 39 tests on ml/, geo/, routers
uv run python -m training.evaluate \
    --data /path/to/ttpla \
    --weights weights/unet_resnet34_ttpla.pth \
    --split val
# → docs/evaluation_results.md  + qualitative panels under docs/screenshots/eval/
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
