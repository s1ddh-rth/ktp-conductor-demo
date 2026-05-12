# Data governance

Where SSEN's data, imagery, and outputs would sit in the regulatory
landscape, and what the prototype already does to demonstrate awareness
of those constraints.

This document is forward-looking — the prototype itself uses public
data (TTPLA, Environment Agency LiDAR, OpenStreetMap) where governance
is straightforward. The point is to map out what changes when the
data becomes operational.

---

## 1. Applicable frameworks

| Framework | Scope | Applicability |
|---|---|---|
| **UK GDPR / Data Protection Act 2018** | Personal data | Aerial imagery may contain identifiable people, vehicles, properties. License plates are personal data. |
| **ESQCR 2002** | Electricity safety | Outputs that feed asset-management decisions are subject to engineering accountability. |
| **NUAR** | Underground assets | National Underground Asset Register is the legal repository for buried services. LV outputs that touch underground records must comply. |
| **IEC 61968** | Utility data interchange | Common Information Model defines shared formats; SSEN's GIS likely conforms. |
| **ISO 27001** | Information security | SSEN's security posture mandates equivalent controls on partner systems. |
| **Cyber Essentials Plus** | UK gov / utility security baseline | Minimum bar for handling sensitive infrastructure data. |

---

## 2. What the prototype does to acknowledge governance

The prototype is a public-data demo, so most controls are theoretical.
The relevant patterns embedded in the code:

- **Configurable retention.** `Settings.upload_retention_hours` controls
  how long uploaded imagery is kept on disk. Default 24 hours;
  configurable per-deployment.
- **No third-party telemetry.** The frontend loads from CDNs (Tailwind,
  Leaflet, Three.js) but the backend has no analytics integrations,
  no error-reporting SaaS, no telemetry beacons.
- **Structured logging without PII.** Logs record request ID, path,
  status, timing — never headers, body content, IP addresses, or
  uploaded image bytes.
- **Rate limiting.** Per-endpoint `slowapi` limits keyed by client IP:
  10 req/min on the GPU-heavy routes (`/api/segment`, `/api/vectorise`,
  `/api/fuse`), 20 req/min on `/api/lidar/sample`, 30 req/min on the
  `/api/infer-hidden/*` deterministic routes. Prevents the demo URL
  being used to mass-process third-party imagery and protects the
  laptop GPU from a viral link.
- **CORS allow-list.** Configurable; defaults to `*` for the demo but
  documented as the first thing to tighten in a production deployment.
- **No authenticated endpoints with persistent state.** All API
  endpoints are stateless beyond the in-memory model. No user
  accounts, no databases of personal data.
- **A `/metrics` endpoint** exposing only operational counters (latency
  histograms, request counts), never request content.

---

## 3. What changes for production at SSEN

The above is the floor. A real deployment would add:

### Data flow

- **At-rest encryption** of uploaded imagery and any cached masks
  (LUKS or equivalent for the disk; per-file encryption for cloud
  storage).
- **In-transit encryption** end-to-end (already provided by
  cloudflared, would need to extend to internal hops).
- **Network isolation** from public internet — the inference service
  would sit behind SSEN's VPN or in a dedicated VPC.

### Personal data handling

- **Auto-blurring** of license plates and faces on imagery ingest, as a
  precaution before any human review (the model itself doesn't need
  these features — they introduce label noise).
- **Documented data-flow diagram** showing where imagery enters the
  system, where it's processed, where outputs go, where logs persist,
  and where (and when) data is deleted.
- **Retention schedule** consistent with SSEN's existing ROPA (Record
  of Processing Activities). For aerial imagery, typical retention is
  the duration of the asset lifecycle (decades) or the duration of the
  operational use case (months).
- **Data subject access** procedure: a way for a data subject to
  request what imagery exists of their property, without creating a
  new attack surface.

### Auditability

- **Append-only audit log** of every model inference: request ID, time,
  inputs hash, outputs hash, model version, user (where authenticated).
  Stored separately from operational logs, with longer retention.
- **Model provenance**: every deployed model traces back to a specific
  training run, dataset version, and code commit. Stored as a model
  card alongside the weights.
- **Reproducibility manifest**: enough information to re-run a specific
  inference exactly, including model weights hash, library versions,
  random seeds, and any post-processing config.

### Access controls

- **Role-based access**: read-only for asset-management staff, write
  for engineers, admin for a small set of operators.
- **Multi-factor authentication** on any administrative interface.
- **Service-account credentials** rotated quarterly.

### Sensitive-data classes

LV-network topology is itself security-sensitive; a publicly-available
map of the LV network would be useful to bad actors. Outputs of the
real system would be classified as **restricted** under SSEN's data
classification scheme, not public.

### Subprocessor controls

- **Cloudflare** (tunnels) — review their data-processing terms.
- **Compute providers** (Colab during research, on-prem for
  production) — model weights and training datasets are SSEN-property,
  must not leave their control without contractual basis.
- **Open-source dependencies** — vulnerability scanning via `uv` or
  `pip-audit` integrated into CI; OSV database for known CVEs.

---

## 4. National Underground Asset Register (NUAR)

NUAR is the UK government's effort to consolidate underground asset
data into a single platform. The Geospatial Commission runs it. From
April 2025 onwards, asset owners are required to share their data with
NUAR; access is limited to authorised users.

For the LV-mapping project, NUAR is relevant in two ways:

1. **Inputs**: LV records that have been digitised and submitted to
   NUAR are a source of ground-truth data we could use for model
   training (with appropriate access agreement).
2. **Outputs**: the predictions our model makes are not authoritative
   records and would not directly populate NUAR. They would inform
   targeted physical surveys, whose results then populate NUAR.

This second point matters because the model's role is **decision
support**, not authoritative record-keeping. The methodology and the
UI should be clear about this — the demo's frontend currently uses
language like "predicted" and "inferred" exactly to maintain that
distinction.

---

## 5. Open data sources used in the prototype

Each of the following is openly licensed; the prototype is in compliance
without further action:

| Source | License | What we use |
|---|---|---|
| TTPLA dataset | Research / open (verify on download) | Training imagery |
| OpenStreetMap | ODbL | Map basemap, building footprints |
| Environment Agency LiDAR composite | OGL v3.0 | Sample LiDAR tile |
| Ordnance Survey Open Zoomstack | OGL v3.0 | Optional UK-specific basemap |

Attribution is provided in the README and in the methodology document.
The prototype code itself is MIT-licensed.
